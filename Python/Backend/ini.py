import streamlit as st
import pandas as pd
from Python.Backend.recup_data import recup_travel, recup_users
from Python.Backend.genV2 import DESTINATIONS
from Model.predict import get_recommendation

def chargement_df():
    # Charger les deux datasets directement depuis le cloud:
    df_users = recup_users()
    df_connexion_users = df_users[["traveler_user_id", "mot_de_passe"]]

    df_travel = recup_travel()

    df_destinations = pd.DataFrame(DESTINATIONS)
    
    return {
        "df_users": df_users, 
        "df_connexion_users": df_connexion_users, 
        "df_destinations":df_destinations, 
        "df_travel": df_travel
        }

def init_session_state():
    df = chargement_df()
    
    if 'premier_affichage' not in st.session_state:
        st.session_state['premier_affichage'] = True
    
    # Initialisation de l'état de connexion
    if 'STATUT_CONNEXION' not in st.session_state:
        st.session_state['STATUT_CONNEXION'] = False
    
    if "app_mode" not in st.session_state:
        st.session_state.app_mode = None
        
    # Initialisation de l'identifiant utilisateur
    if 'df_users' not in st.session_state:
        st.session_state['df_users'] = df["df_users"]
        
    # Initialisation de l'identifiant utilisateur
    if 'df_connexion_users' not in st.session_state:
        st.session_state['df_connexion_users'] = df["df_connexion_users"]
        
    # Initialisation de l'identifiant utilisateur
    if 'df_travel' not in st.session_state:
        st.session_state['df_travel'] = df["df_travel"]
        
    # Initialisation de l'identifiant utilisateur
    if 'df_destinations' not in st.session_state:
        st.session_state['df_destinations'] = df["df_destinations"]
        
    # Initialisation de l'identifiant utilisateur
    if 'user' not in st.session_state:
        st.session_state['user'] = None
        
    if 'chemin_image' not in st.session_state:
        st.session_state['chemin_image'] = "https://raw.githubusercontent.com/vincentmori/Image_ville_projet_rec_sys/refs/heads/main/"
        
def init_user(user_id):
    st.session_state['STATUT_CONNEXION'] = True 
    
    df_users = st.session_state['df_users']
    
    mask_user = df_users["traveler_user_id"] == user_id
    
    user = df_users[mask_user]
    
    st.session_state["user"] = user
    
    df_travel = st.session_state['df_travel']
    
    mask_histo_user = df_travel["User ID"] == user_id
    
    historique_user = df_travel[mask_histo_user]
    
    historique_user = historique_user.reset_index(drop=True)
    
    historique_user["Start date"] = pd.to_datetime(historique_user["Start date"])
    
    st.session_state["historique_user"] =  historique_user.sort_values(
        by="Start date",
        ascending=False
    ).reset_index(drop=True)

    
    reco_user(user_id)
    
def reco_user(user_id):
    reco_user = get_recommendation(user_id)

    st.session_state["reco_user"] = reco_user
    