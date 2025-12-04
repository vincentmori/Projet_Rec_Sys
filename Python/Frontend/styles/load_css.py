import streamlit as st
from pathlib import Path

# ---------------------------
# FONCTION DE CHARGEMENT DU STYLES DEFINI
# ---------------------------
def load_css():
    """
    Load and inject the content of the css file
    """
    # Construire le chemin complet vers styles.css
    css_file_path = Path("Python/Frontend/styles/styles.css")
    
    try:
        # Ouvrir le fichier et injecter le contenu
        with open(css_file_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        # Afficher une erreur claire si le fichier n'est pas trouvé
        st.error(f"Error: CSS FILE NOT FOUND : {css_file_path}")
    except Exception as e:
        st.error(f"ERROR WHILE LOADING CSS FILE : {e}")