import streamlit as st
from time import sleep    
from Python.Frontend.styles.load_css import load_css
from Python.Backend.ini import init_session_state
from Python.Backend.connexion import auto_login

st.set_page_config(page_title="TripplyBuddy", page_icon="🌍", layout="wide")

# -------------------------------
# INITIALIZATION SESSION STATE
# -------------------------------
init_session_state()

load_css()

with st.spinner("⏳ Auto Connection..."):
    if auto_login():
        st.success(f"Connexion Succeeded! Traveler name: {st.session_state['user']['traveler_name'].loc[0]}")
        sleep(0.5)

# Redirection vers la page Accueil
st.switch_page("pages/1_Accueil.py")