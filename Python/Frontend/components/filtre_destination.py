import streamlit as st
from Python.Frontend.components.card import get_all_cards_html, get_all_cards_histo_html

def affichage_card(df, histo=False):  
    if not histo:  
        cards_html = get_all_cards_html(df)
    else:
        cards_html = get_all_cards_histo_html(df)
    
    final_html = f"""
    <div class="scrollable-container">
        {cards_html}
    </div>
    """
    
    st.markdown(final_html, unsafe_allow_html=True)  