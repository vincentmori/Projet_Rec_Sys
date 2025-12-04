import streamlit as st 
import os  
import pandas as pd
# Importation corrigée :
from datetime import date, timedelta
from Python.Backend.ini import reco_user
from Model.predict import reset_predictor
from time import sleep
from Python.Backend.write import update_travel_table

SESSION_FILE = os.path.join("Data", "rester_connecter.txt")

@st.dialog("Add Travel", width='large')
def add_travel():
    # --- Récupération des données de session ---
    df_user = st.session_state['df_users']
    user_df = st.session_state['user']
    df_histo = st.session_state['historique_user']
    df_travel = st.session_state['df_travel']
    df_dest = st.session_state['df_destinations']
    
    # Extraction des infos utilisateur (pour éviter les erreurs d'alignement Pandas)
    # On utilise .iloc[0] ou .item() car 'user_df' est un DataFrame d'une seule ligne
    user_id = user_df["traveler_user_id"].iloc[0]
    user_name = user_df["traveler_name"].iloc[0]
    user_age = user_df["traveler_age"].iloc[0]
    user_gender = user_df["traveler_gender"].iloc[0]
    user_nationality = user_df["traveler_nationality"].iloc[0]
    
    
    st.title("Add a previous or futur travel:")
    
    col_city, col_start, col_end = st.columns(3)
    
    with col_city:
        city = st.text_input("City")
        
    date_aujourdhui = date.today()
    date_max = date_aujourdhui + timedelta(days=365) # Max dans un an
    
    date_hier = date_aujourdhui - timedelta(days=1)
        
    with col_start:
        start_date = st.date_input(
            "Start Date",
            value=date_hier, 
            max_value=date_max,
            help="Date de début du voyage (peut être passée ou future)."
        )
        
    with col_end:
        end_date = st.date_input(
            "End date",
            value=date_aujourdhui,
            min_value=start_date,    # Date de fin doit être >= Date de début
            max_value=date_max,
            help="Date de fin du voyage."
        )
        
    st.markdown("---")
    
    col_accomodation, col_transport = st.columns(2)
    
    list_accomodation = sorted(list(df_travel["Accommodation type"].unique()))
    
    all_transport = df_user["transport_core_modes"].astype(str).str.cat(sep=';')
    
    list_transport_modes = pd.Series(all_transport.split(';')).str.strip().unique().tolist()
    
    with col_accomodation:
        accomodation = st.selectbox("Accommodation type", list_accomodation, index=0)
    
    with col_transport:
        new_transport = st.selectbox("Local transport mode", list_transport_modes)
        
    st.markdown("---")
    
    col_acc_cost, col_cost = st.columns(2)
    
    with col_acc_cost:
        acc_cost = st.number_input(
            "Accommodation cost (€)", 
            min_value=0.0,
            value=50.0,
            help="Coût total de l'hébergement pour toute la durée du voyage."
        )
    
    # Ajustement de la valeur par défaut pour total_cost pour être >= acc_cost
    default_total_cost = max(50.0, acc_cost)
    
    with col_cost:
        total_cost = st.number_input(
            "Total cost (€)", 
            min_value=acc_cost, # Le coût total doit être au moins égal au coût d'hébergement
            value=default_total_cost + 50.0, # Suggestion d'un coût total supérieur
            help="Coût total du voyage (hébergement + autres dépenses)."
        )
        
    st.markdown("---")
    
    button, _, quit = st.columns([3, 7, 2])
        
    with button:
        if st.button("Register and add travel", key="create_account_btn"):
            if not city.strip() or acc_cost is None or total_cost is None:
                st.error("Please fill in the city and all cost fields.")
            
            # --- 1. Récupérer les infos de destination ---
            else:
                mask_info_city = df_dest["city"].str.lower() == city.lower()
                info_city = df_dest[mask_info_city]
                
                if info_city.empty:
                    st.error(f"City '{city}' not found in the destinations list. Please check the spelling.")
                    return # Arrêter l'exécution si la ville n'est pas trouvée

                # Récupérer le pays et former la destination (ex: Paris, France)
                country = info_city["country"].iloc[0]
                destination = f"{city}, {country}"
                
                # --- 2. Calculs et Nouveaux IDs ---
                Duration = end_date - start_date
                number_of_days = Duration.days
                
                average_daily_cost = total_cost - acc_cost # Coût Total Hors Hébergement
                
                last_id_trip = df_travel.tail(1)["Trip ID"].iloc[0]
                number_part_tripID = last_id_trip[1:]
                last_id_nb = int(number_part_tripID)
                
                new_trip_id_nb = last_id_nb + 1
                new_trip_id = f"T{new_trip_id_nb:05}" 
                
                # --- 3. Création et Ajout du DataFrame ---
                new_histo = pd.DataFrame([{
                    "Trip ID": new_trip_id, 
                    "User ID": user_id, 
                    "Destination": destination, 
                    "Start date": start_date,
                    "End date": end_date, 
                    "Duration (days)": number_of_days,
                    "Traveler name": user_name, 
                    "Traveler age": user_age, 
                    "Traveler gender": user_gender, 
                    "Traveler nationality": user_nationality,
                    "Accommodation type": accomodation, 
                    "Accommodation cost": acc_cost,
                    "Average Daily Cost": average_daily_cost, 
                    "Total cost": total_cost,
                    "Local Transport Mode": new_transport
                }]) 
                
                # Mise à jour des DataFrames en Session
                df_add_histo = pd.concat([df_histo, new_histo], ignore_index=True)
                df_add_histo["Start date"] = pd.to_datetime(df_add_histo["Start date"])
                
                # Tri du nouvel historique (plus récent au plus ancien)
                st.session_state["historique_user"] = df_add_histo.sort_values(
                    by="Start date",
                    ascending=False
                ).reset_index(drop=True)
                
                st.session_state["df_travel"] = pd.concat([df_travel, new_histo], ignore_index=True)
                
                # --- 4. Persistance et Mise à jour des Reco ---
                with st.spinner("⏳ Adding your new data to our database..."):
                    ajout_bdd = update_travel_table(st.session_state["df_travel"])
                
                if ajout_bdd:
                    st.success("Travel added with success to the database!")
                else:
                    st.error("Error adding travel to the database.")
                    
                sleep(0.5)
                
                reset_predictor()
                with st.spinner("⏳ Updating your recommendations..."):
                    reco_user(user_id) 
                
                st.rerun() 
    
    with quit:
        if st.button("Quit", key="back_to_login_btn"):
            st.rerun()