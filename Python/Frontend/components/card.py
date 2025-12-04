import streamlit as st
import unicodedata
import re 
import pandas as pd 

# ------------------------------
# ENLEVE LES ACCENTS DES VILLES
# ------------------------------
def remove_accents(text):
    """
    remove accents for cities such as Malé.
    """
    normalized_text = unicodedata.normalize('NFD', text)
    
    cleaned_text = re.sub(r'[\u0300-\u036f]', '', normalized_text)
    
    return cleaned_text

# --------------------------------------------------------
# CREATION D'UNE CARTE POUR UNE VILLE
# AFFICHAGE DES DATES SI C'EST UNE CARTE PREVIOUS TRAVELS
# --------------------------------------------------------
def get_city_card_html(city, Start_date=None, End_date=None, Histo=False):
    df_destinations = st.session_state["df_destinations"]
    
    mask_city = df_destinations["city"] == city
    info_city = df_destinations[mask_city]
    
    if info_city.empty:
        return f""
        
    country = info_city["country"].iloc[0]
    
    image_url = f"{st.session_state['chemin_image']}{remove_accents(city)}.jpg"
    
    if not Histo: 
        card_html = f"""
<div class="destination-card-v2">
    <img src="{image_url}" class="card-v2-image" alt="{city}, {country}">
    <div class="card-v2-info">
        <p class="card-v2-city-country">
            {city}, {country}
        </p>
    </div>
</div>
"""
    else: 
        date_histo = f"From {Start_date} to {End_date}"
        card_html = f"""
<div class="destination-card-v2">
    <img src="{image_url}" class="card-v2-image" alt="{city}, {country}">
    <div class="card-v2-info">
        <p class="card-v2-city-country">
            {city}, {country}
        </p>
        <p class="card-v2-dates">{date_histo}</p> 
    </div>
</div>
        """
    
    return card_html
    
# ------------------------------------------
# RECUPERE TOUTES LES CARTES D'UN DATAFRAME
# ------------------------------------------
def get_all_cards_html(df):
    all_cards_html = []
    
    for city in df["city"]:
        card_html = get_city_card_html(city)
        all_cards_html.append(card_html)
        
        
    return "".join(all_cards_html)

# -----------------------------------------------------
# RECUPERE TOUTES LES CARTES HISTORIQUE Du DATAFRAME
# -----------------------------------------------------
def get_all_cards_histo_html(df):
    all_cards_html = []
    
    df["Start date"] = pd.to_datetime(df["Start date"], format="%Y-%m-%d", errors='coerce')
    df["End date"] = pd.to_datetime(df["End date"], format="%Y-%m-%d", errors='coerce')
    
    for _, row in df.iterrows():
        Start_date = row["Start date"].strftime("%d/%m/%Y")   
        End_date = row["End date"].strftime("%d/%m/%Y")
        
        card_html = get_city_card_html(
                    row["city"], 
                    Start_date=Start_date, 
                    End_date=End_date, 
                    Histo=True
                )

        all_cards_html.append(card_html)
        
    return "".join(all_cards_html)
