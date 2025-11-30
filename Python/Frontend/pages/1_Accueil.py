import streamlit as st
from components.header import display_header
from components.footer import display_footer
from components.card import get_all_cards_html
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
    Titre_html = '<span id="title-page">Accueil</span>'
    st.markdown(f"{Titre_html}", unsafe_allow_html=True)
    
    sous_titre_html = '<span id="subtitle-page">Our Destinations</span>'
    st.markdown(f"{sous_titre_html}", unsafe_allow_html=True)
    
    df_destinations = st.session_state["df_destinations"]
    
    cards_html = get_all_cards_html(df_destinations)
    
    final_html = f"""
    <div class="scrollable-container">
        {cards_html}
    </div>
    """
    
    st.markdown(final_html, unsafe_allow_html=True)    
    
content_accueil()

# -------------------------------
# FOOTER
# -------------------------------
display_footer()