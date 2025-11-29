import streamlit as st
from components.header import display_header
from components.footer import display_footer
from styles.load_css import load_css
from components.login_dialog import login_dialog
import os 

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
        # Ajout d'espace vertical pour aligner le bouton avec le titre
        st.write("") 
        st.write("") 
        
        # 2. Le bouton "Se déconnecter"
        if st.button("Se déconnecter", key="logout_compte", use_container_width=True):
            # 3. Appel de la fonction de déconnexion
            logout_user()
            

def content_compte():
    st.title("Mon Compte")

if st.session_state["STATUT_CONNEXION"]:
    content_compte_connecte()
else:
    content_compte()
    login_dialog()

# -------------------------------
# FOOTER
# -------------------------------
display_footer()