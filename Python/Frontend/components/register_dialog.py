import streamlit as st  
import os   
import pandas as pd
from Python.Backend.connexion import check_connexion
from Python.Backend.ini import init_user
from time import sleep

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..', '..')) 

SESSION_FILE = os.path.join(PROJECT_ROOT, "Data", "rester_connecter.txt")

@st.dialog("Register", width='large')
def register_dialog():
    df_user = st.session_state['df_users']
    
    st.title("Create an account:")
    """Name, Age and gender"""
    col_name, col_age, col_gender = st.columns(3)
    
    with col_name:
        new_username = st.text_input("Name")
        
    with col_age:
        new_age = st.text_input("Age")
        
    with col_gender:
        new_gender = st.selectbox("Gender", ["Male", "Female"], index=0)
        
    """Nationality and continent"""
    list_nationality = sorted(list(df_user["traveler_nationality"].unique()))
    list_continent = sorted(list(df_user["traveler_continent"].unique()))
    
    col_nationality, col_continent = st.columns(2)
    
    with col_nationality:
        new_nationality = st.selectbox("Nationality", list_nationality, index=0)
        
    with col_continent:
        new_continent = st.selectbox("Continent", list_continent, index=0)
        
    """Climate and destination type """
    list_profile = sorted(list(df_user["profile_type"].unique()))
    list_climat = sorted(list(df_user["climate_pref"].unique()))
    list_dest_type = sorted(list(df_user["primary_dest_type"].unique()))
    
    col_profile, col_climat, col_dest_type = st.columns(3)
    
    with col_profile:
        new_profile_type = st.selectbox("Favorite vacation type", list_profile, index=0)
        
    with col_climat:
        new_climat_type = st.selectbox("Favorite climat type", list_climat, index=0)
        
    with col_dest_type:
        new_dest_type = st.selectbox("Main destination type", list_dest_type, index=0)
        
    """Accomodation and transport mode"""
    list_accomodation = sorted(list(df_user["acc_pref"].unique()))
    
    all_transport = df_user["transport_core_modes"].astype(str).str.cat(sep=';')
    
    list_transport_modes = pd.Series(all_transport.split(';')).str.strip().unique().tolist()
    
    # Clean empty values
    list_transport_modes = [mode for mode in list_transport_modes if mode]
    
    col_accomodation, col_transport = st.columns(2)
    
    with col_accomodation:
        new_accomodation = st.selectbox("Favorite accomodation type", list_accomodation, index=0)
        
    with col_transport:
        list_new_transport = st.multiselect("Favorite transports", list_transport_modes, default=[])
    
    new_transport = ";".join(list_new_transport)
        
    """Password and remember me"""
    col_user_id, col_password = st.columns(2)
    
    with col_password:
        new_password = st.text_input("New Password", type="password")
        
    with col_user_id:
        new_user_id = st.text_input("ID")
        
    """Connect button"""
    button, _, col_remember_me = st.columns([3, 7, 2])
    with col_remember_me:
        remember_me = st.checkbox("Stay login")
        
    with button:
        if st.button("Register and login", key="create_account_btn"):
            if new_username.strip() == '' or new_age.strip() == '' or new_password.strip() == '' or new_user_id.strip() == '':
                st.error("Please enter all values")
            elif new_user_id.strip() in list(df_user["traveler_user_id"]):
                st.error("ID already used. Please enter another one")
            else:                
                new_user = pd.DataFrame({
                    "traveler_user_id": [new_user_id], "traveler_name": [new_username], 
                    "traveler_age": [new_age], "traveler_gender": [new_gender], 
                    "traveler_nationality": [new_nationality], "profile_type": [new_profile_type],
                    "climate_pref": [new_climat_type], "primary_dest_type": [new_dest_type], 
                    "acc_pref": [new_accomodation], "transport_core_modes": [new_transport], 
                    "traveler_continent": [new_continent], "mot_de_passe": [new_password]
                })
                
                st.session_state["df_users"] = pd.concat([df_user, new_user], ignore_index=True)
                
                st.session_state["df_connexion_users"] = pd.concat([st.session_state["df_connexion_users"], new_user[["traveler_user_id", "mot_de_passe"]]], 
                                                                    ignore_index=True)
                        
                st.success("Acount created with success.")
                
                check_co, message_erreur = check_connexion(new_user_id, new_password)
                
                if not check_co:
                    st.error(message_erreur)
                else:
                    init_user(new_user_id)
                    st.success(f"Connexion Succeeded! Traveler name: {st.session_state['user']['traveler_name'].iloc[0]}")
                    
                    if remember_me:
                        try:
                            with open(SESSION_FILE, "w") as f:
                                f.write(f"{new_user_id}|{new_password}")
                        except Exception as e:
                            print(f"Impossible to write: {e}")
                    else:
                        if os.path.exists(SESSION_FILE):
                            os.remove(SESSION_FILE)
                    
                    sleep(0.5)
                    
                    st.rerun()
    
    if st.button("Already registered ? Go back to connection", key="back_to_login_btn"):
        st.session_state["STATUT_CONNEXION"] = False
        st.session_state.app_mode = None
        st.rerun()
    