import streamlit as st
from Python.Frontend.components.header import display_header
from Python.Frontend.components.footer import display_footer
from Python.Frontend.styles.load_css import load_css
from Python.Frontend.components.login_dialog import login_dialog
from Python.Frontend.components.register_dialog import register_dialog
from Python.Frontend.components.filtre_destination import affichage_card
import os 
import pandas as pd

st.set_page_config(page_title="TripplyBuddy", page_icon="🌍", layout="wide")

# -------------------------------
# LOG OUT FUNCTION
# -------------------------------
def logout_user():
    st.session_state["STATUT_CONNEXION"] = False
    
    PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..', '..')) 
    SESSION_FILE = os.path.join(PROJECT_ROOT, "Data", "rester_connecter.txt")
    if os.path.exists(SESSION_FILE):
        os.remove(SESSION_FILE)

    st.rerun() 

# -------------------------------
# MISE EN FROME RECOMMANDATION
# -------------------------------    
def mise_en_forme_df(data):
    structured_data = []

    for i, item in enumerate(data):
        # 2. split ',' pour avoir la ville puis le pays
        parts = item.split(",")

        # Assurez-vous d'avoir au moins deux parties (Ville et Pays)
        if len(parts) == 2:
            city = parts[0].strip()
            country = parts[1].strip()
        else:
            city = parts[0].strip()
            country = None

        structured_data.append({
            "index": i,
            "city": city,
            "country": country
        })
        
    return pd.DataFrame(structured_data)

# -------------------------------
# CHARGEMENT DU CSS & HEADER
# -------------------------------
load_css()
display_header()

# -------------------------------
# CONTENT
# -------------------------------
def content_compte_connecte():
    col_title, col_button = st.columns([0.8, 0.2])

    with col_title:
        st.title("Mon compte")

    with col_button:
        st.write("") 
        st.write("") 
        
        if st.button("Se déconnecter", key="logout_compte", use_container_width=True):
            # 3. Appel de la fonction de déconnexion
            logout_user()
            
    st.header("Your recommandations")
    df_reco_user = st.session_state["reco_user"]
    df_reco_affiche = mise_en_forme_df(st.session_state["reco_user"])
    affichage_card(df_reco_affiche)
    
    if not st.session_state["historique_user"].empty:
        st.header("Your previous travels")
        df_histo_user = st.session_state["historique_user"]
        df_city_histo = mise_en_forme_df(df_histo_user["Destination"])
        affichage_card(df_city_histo)
    else:
        st.header("No previous travels")
            

def content_compte():
    st.title("Mon Compte")

if st.session_state["STATUT_CONNEXION"]:
    content_compte_connecte()
elif st.session_state.app_mode == 'register':
    content_compte()
    register_dialog()
else:
    content_compte()
    login_dialog()

# -------------------------------
# FOOTER
# -------------------------------
display_footer()