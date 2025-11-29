import streamlit as st
import pandas as pd
from Python.Backend.recup_data import recup_travel, recup_users
from Python.Backend.genV2 import DESTINATIONS
# from Model.predict import get_recommendation

def chargement_df():
    # Charger les deux datasets directement depuis le cloud:
    df_users = recup_users()
    df_connexion_users = pd.DataFrame({
        "user_id": df_users["user_id"],
        "mdp": df_users["user_id"]
    })

    df_travel = recup_travel()

    df_destinations = DESTINATIONS
    
    return {
        "df_users": df_users, 
        "df_connexion_users": df_connexion_users, 
        "df_destinations":df_destinations, 
        "df_travel": df_travel
        }

df = chargement_df()

def init_session_state():
    
    # Initialisation de l'état de connexion
    if 'STATUT_CONNEXION' not in st.session_state:
        st.session_state['STATUT_CONNEXION'] = False
        
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
        
def init_user(user_id):
    st.session_state['STATUT_CONNEXION'] = True 
    
    df_users = df["df_users"]
    
    mask_user = df_users["user_id"] == user_id
    
    user = df_users[mask_user]

    st.session_state["user"] = user
    
    """reco_user = get_recommendation(user_id)

    st.session_state["reco_user"] = reco_user"""
    
    
    
    