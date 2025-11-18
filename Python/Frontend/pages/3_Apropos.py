import streamlit as st
from styles.load_css import load_css
from components.header import display_header
from components.footer import display_footer

st.set_page_config(page_title="TripplyBuddy", page_icon="🌍", layout="wide")

# -------------------------------
# CHARGEMENT DU CSS & HEADER
# -------------------------------
load_css()
display_header()

# -------------------------------
# CONTENT
# -------------------------------

st.title("À propos du projet 🎓")

st.write("""
Bienvenue sur TripplyBuddy ! Ce projet a été réalisé dans le cadre d'un projet étudiant 
sur les systèmes de recommandation personnalisée dans le domaine du voyage.

### Objectif
Fournir des recommandations de destinations, hôtels et activités adaptées aux préférences des utilisateurs, 
en combinant machine learning et UX moderne.

### Contexte
- Projet académique : système de recommandation  
- Données : Kaggle Traveler Trip Dataset, ~2000 records  
- Futur : intégrer feedback utilisateur réel et API cloud  

### Équipe
- Étudiant 1  
- Étudiant 2  
- Étudiant 3  

Nous avons conçu cette application pour montrer comment un système de recommandation personnalisé peut aider à planifier des voyages facilement et efficacement.
""")

# -------------------------------
# FOOTER
# -------------------------------

display_footer()