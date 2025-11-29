import streamlit as st
import os
import sys
from time import sleep

project_root = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))

# Ajouter le chemin racine à sys.path
if project_root not in sys.path:
    sys.path.append(project_root)
    
    
from Python.Frontend.styles.load_css import load_css
from Python.Backend.ini import init_session_state
from Python.Backend.connexion import auto_login

st.set_page_config(page_title="TripplyBuddy", page_icon="🌍", layout="wide")

# -------------------------------
# INITIALIZATION SESSION STATE
# -------------------------------
init_session_state()

load_css()

if auto_login():
    st.success(f"Connexion Succeeded! Traveler name: {st.session_state['user']['traveler_name'].loc[0]}")
    sleep(0.5)

# Redirection vers la page Accueil
st.switch_page("pages/1_Accueil.py")