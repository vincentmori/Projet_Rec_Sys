import streamlit as st
from Python.Frontend.components.header import display_header
from Python.Frontend.components.footer import display_footer
from Python.Frontend.styles.load_css import load_css
from Python.Frontend.components.login_dialog import login_dialog
from Python.Frontend.components.register_dialog import register_dialog
from Python.Frontend.components.filtre_destination import affichage_card
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
        st.write("") 
        st.write("") 
        
        if st.button("Se déconnecter", key="logout_compte", use_container_width=True):
            # 3. Appel de la fonction de déconnexion
            logout_user()
            
    st.header("Your recommandations")
    
    affichage_card(st.session_state["reco_user"])
    
    if not st.session_state["historique_user"].empty():
        st.header("Your previous travels")
        affichage_card(st.session_state["historique_user"])
            

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