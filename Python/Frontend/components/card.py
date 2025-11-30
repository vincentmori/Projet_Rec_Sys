import streamlit as st
import unicodedata
import re 

def remove_accents(text):
    """
    remove accents for cities such as Malé.
    """
    normalized_text = unicodedata.normalize('NFD', text)
    
    cleaned_text = re.sub(r'[\u0300-\u036f]', '', normalized_text)
    
    return cleaned_text

def get_city_card_html(city):
    df_destinations = st.session_state["df_destinations"]
    
    mask_city = df_destinations["city"] == city
    info_city = df_destinations[mask_city]
    
    if info_city.empty:
        return f""
        
    country = info_city["country"].iloc[0]
    
    image_url = f"https://raw.githubusercontent.com/vincentmori/Image_ville_projet_rec_sys/refs/heads/main/{remove_accents(city)}.jpg"
    
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
    return card_html
    
def get_all_cards_html(df_city):
    
    all_cards_html = []
    
    for ville in df_city["city"]:
        card_html = get_city_card_html(ville)
        all_cards_html.append(card_html)
            
    return "".join(all_cards_html)
