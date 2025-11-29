import streamlit as st  
import os   
import sys
from Python.Backend.connexion import check_connexion
from Python.Backend.ini import init_user
from time import sleep

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..', '..')) 

SESSION_FILE = os.path.join(PROJECT_ROOT, "Data", "rester_connecter.txt")

@st.dialog("Connexion")
def login_dialog():   
    # st.dialog utilise des widgets Streamlit normaux à l'intérieur
    user_id = st.text_input("Identifiant")
    password = st.text_input("Mot de passe", type='password')
    remember_me = st.checkbox("Rester connecté")

    if st.button("Se connecter", use_container_width=True):
        if not user_id.strip() or not password.strip():
            st.error("Please enter your ID and Password.")
        else:
            check_co, message_erreur = check_connexion(user_id, password)
            
            if not check_co:
                st.error(message_erreur)
            else:
                init_user(user_id)
                st.success(f"Connexion Succeeded! Traveler name: {st.session_state['user']['traveler_name'].loc[0]}")
                
                if remember_me:
                    try:
                        with open(SESSION_FILE, "w") as f:
                            f.write(f"{user_id}|{password}")
                    except Exception as e:
                        print(f"Impossible to write: {e}")
                        st.session_state['STATUT_CONNEXION'] = False
                else:
                    if os.path.exists(SESSION_FILE):
                        os.remove(SESSION_FILE)
                
                sleep(0.5)
                
                st.switch_page("pages/1_Accueil.py")