import streamlit as st
from components.header import display_header
from components.footer import display_footer
from styles.load_css import load_css

st.set_page_config(page_title="TripplyBuddy", page_icon="🌍", layout="wide")

load_css()
display_header()

# Redirection vers la page Accueil
st.switch_page("pages/1_Accueil.py")