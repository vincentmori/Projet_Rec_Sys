import streamlit as st

def load_css():
    with open("Python/Frontend/styles/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)