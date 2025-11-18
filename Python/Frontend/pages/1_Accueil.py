import streamlit as st
from components.header import display_header
from components.footer import display_footer
from styles.load_css import load_css

st.set_page_config(page_title="TripplyBuddy", page_icon="🌍", layout="wide")

# -------------------------------
# CHARGEMENT DU CSS & HEADER
# -------------------------------
load_css()
display_header()

# -------------------------------
# CONTENT
# -------------------------------
st.title("Bienvenue sur TripplyBuddy ✈️")
st.write("Page d'accueil")

# -------------------------------
# FOOTER
# -------------------------------
display_footer()