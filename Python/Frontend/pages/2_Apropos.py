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
def content_Apropos():
    logo_html_inline = '<span id="logo-txt">TripplyBuddy</span>'

    st.title("About the Project 🎓")

    st.markdown(f"""
    Welcome to {logo_html_inline}! This project was developed as part of a **student initiative** focused on **personalized recommendation systems** within the travel industry.
    """, unsafe_allow_html=True) 

    st.markdown('<div class="section-title">Objective</div>', unsafe_allow_html=True)
    st.write("""
    Our goal is to provide **tailored travel recommendations**, including specific destinations 
    and accommodation suggestions, based on the unique profile and preferences of each user.
    """)

content_Apropos()
# -------------------------------
# FOOTER
# -------------------------------

display_footer()