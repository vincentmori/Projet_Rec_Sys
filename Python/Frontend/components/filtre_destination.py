import streamlit as st
from Python.Frontend.components.card import get_all_cards_html

def affichage_card(df):    
    cards_html = get_all_cards_html(df)
    
    final_html = f"""
    <div class="scrollable-container">
        {cards_html}
    </div>
    """
    
    st.markdown(final_html, unsafe_allow_html=True)  