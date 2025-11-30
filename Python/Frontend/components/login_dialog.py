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
    user_id = st.text_input("Id")
    password = st.text_input("Password", type='password')
    
    col1, _, col2 = st.columns([2, 3, 2])
    with col2:
        remember_me = st.checkbox("Stay login")

    with col1:
        if st.button("Register", key="to_register_btn"):
            st.session_state.app_mode = 'register' # Changement d'état
            st.rerun()

    if st.button("Connect", use_container_width=True):
        if not user_id.strip() or not password.strip():
            st.error("Please enter your ID and Password.")
        else:
            check_co, message_erreur = check_connexion(user_id, password)
            
            if not check_co:
                st.error(message_erreur)
            else:
                init_user(user_id)
                st.success(f"Connexion Succeeded! Traveler name: {st.session_state['user']['traveler_name'].iloc[0]}")
                
                if remember_me:
                    try:
                        with open(SESSION_FILE, "w") as f:
                            f.write(f"{user_id}|{password}")
                    except Exception as e:
                        print(f"Impossible to write: {e}")
                else:
                    if os.path.exists(SESSION_FILE):
                        os.remove(SESSION_FILE)
                
                sleep(0.5)
                
                st.rerun()