import streamlit as st
from Python.Frontend.components.header import display_header
from Python.Frontend.components.footer import display_footer
from Python.Frontend.styles.load_css import load_css
from Python.Frontend.components.filtre_destination import affichage_destination

st.set_page_config(page_title="TripplyBuddy", page_icon="🌍", layout="wide")

# -------------------------------
# CHARGEMENT DU CSS & HEADER
# -------------------------------
load_css()
display_header()

# -------------------------------
# CONTENT
# -------------------------------
def content_accueil():
    st.title("Accueil")
    
    st.header("Our destinations")
    
    affichage_destination()
    
  
    
content_accueil()

# -------------------------------
# FOOTER
# -------------------------------
display_footer()