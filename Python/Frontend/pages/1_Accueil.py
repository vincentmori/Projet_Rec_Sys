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
def content_accueil():
    st.title("Bienvenue sur TripplyBuddy ✈️")
    st.write("Page d'accueil")
    
    """if st.session_state['STATUT_CONNEXION']:
        st.write(st.session_state["reco_user"])"""
    
content_accueil()

# -------------------------------
# FOOTER
# -------------------------------
display_footer()