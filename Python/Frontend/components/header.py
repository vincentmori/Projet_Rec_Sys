import streamlit as st

def display_header():
    st.markdown("""
    <div id="custom-header">
        <div id="logo">TripplyBuddy</div>
        <div class="header-menu">
            <a href="/Accueil" target="_self">Accueil</a>
            <a href="/Connexion" target="_self">Connexion</a>
            <a href="/Apropos" target="_self">À propos</a>
        </div>
    </div>
    """, unsafe_allow_html=True)