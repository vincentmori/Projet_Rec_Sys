import streamlit as st
import os 
import sys
from Python.Backend.ini import init_user

def check_connexion(user_id, mdp):
    if 'df_connexion_users' in st.session_state:
        df_connexion_users = st.session_state["df_connexion_users"]
    
    check = False
    erreur = None
    if user_id in list(df_connexion_users["user_id"]):
        mask_user = df_connexion_users["user_id"] == user_id
        mdp_valide = df_connexion_users[mask_user]["mdp"]
        
        if (mdp == mdp_valide.loc[0]):
            check = True 
        else:
            erreur = "Connexion Failed: Invalid Password"
    else:
        erreur = "Connexion Failed: Invalid User ID"
        
    return check, erreur

def auto_login():
    """If user already connected."""
    if st.session_state['STATUT_CONNEXION']:
        return
    
    PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..', '..')) 
    SESSION_FILE = os.path.join(PROJECT_ROOT, "Data", "rester_connecter.txt")

    if os.path.exists(SESSION_FILE):
        try:
            with open(SESSION_FILE, "r") as f:
                content = f.read().strip()
                print(content)
            
            if content and '|' in content:
                saved_user_id, saved_password = content.split('|')
                
                check_co, _ = check_connexion(saved_user_id, saved_password)
                
                if check_co:
                    init_user(saved_user_id) 
                    st.session_state['STATUT_CONNEXION'] = True
                    return True 
                else:
                    os.remove(SESSION_FILE)
            
        except Exception as e:
            print(f"Erreur de lecture du fichier de session : {e}")
            if os.path.exists(SESSION_FILE):
                os.remove(SESSION_FILE)
                
    return False