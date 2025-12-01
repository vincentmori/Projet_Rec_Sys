import streamlit as st
from Python.Frontend.components.card import get_all_cards_html

def affichage_destination():
    df_destinations = st.session_state["df_destinations"]
    
    cards_html = get_all_cards_html(df_destinations)
    
    final_html = f"""
    <div class="scrollable-container">
        {cards_html}
    </div>
    """
    
    st.markdown(final_html, unsafe_allow_html=True)  