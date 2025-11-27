import pandas as pd
import numpy as np
from datetime import date, timedelta
import random

# ============================================================
#    GEN.PY — Version structurée SANS dictionnaires
#    Les dictionnaires seront fournis dans le message suivant.
# ============================================================

# ------------------------------------------------------------
# PARAMÈTRES GÉNÉRAUX
# ------------------------------------------------------------

NB_USERS = 1500              # nombre d'utilisateurs à générer
START_YEAR = 2015
END_YEAR = 2025

# Coûts journaliers généraux selon niveau
DAILY_COST_MAPPING = {
    'Low': 45.00,
    'Medium': 90.00,
    'High': 180.00
}

# ------------------------------------------------------------
# 🚨 DICTIONNAIRES VIDES (À REMPLIR DANS MESSAGE 2)
# ------------------------------------------------------------


ACCOMMODATION_TYPES = ['Hotel', 'Hostel', 'Airbnb', 'Resort', 'Villa', 'Ryokan', 'Camping', 'Guesthouse']
NATIONALITIES = [
    'French', 'German', 'American', 'Brazilian', 'Japanese', 'Australian', 'South African',
    'Indian', 'Nigerian', 'Mexican', 'South Korean', 'Spanish', 'Egyptian', 'Italian', 
    'Canadian', 'Vietnamese', 'Colombian', 'Russian', 'Saudi', 'Swedish'
]
COHERENCE_MAPPING = {
    'French': {'Continent': 'Europe', 'M': ['Marc', 'Pierre', 'Thomas', 'Hugo', 'Simon', 'Alexandre', 'Louis', 'Antoine', 'Lucas', 'Gabriel'], 'F': ['Sophie', 'Marie', 'Camille', 'Louise', 'Alice', 'Jeanne', 'Emma', 'Chloé', 'Inès', 'Manon']},
    'German': {'Continent': 'Europe', 'M': ['Klaus', 'Hans', 'Ludwig', 'Felix', 'Jonas', 'Maximilian', 'Elias', 'Leon', 'Paul', 'Noah'], 'F': ['Greta', 'Heidi', 'Ingrid', 'Lena', 'Hanna', 'Anja', 'Mia', 'Emilia', 'Frieda', 'Clara']},
    'American': {'Continent': 'North America', 'M': ['David', 'James', 'Michael', 'Ethan', 'Jacob', 'William', 'Alex', 'Ryan', 'Sam', 'Chris'], 'F': ['Emily', 'Jessica', 'Sarah', 'Olivia', 'Ava', 'Sophia', 'Isabella', 'Charlotte', 'Amelia', 'Evelyn']},
    'Brazilian': {'Continent': 'South America', 'M': ['João', 'Lucas', 'Rafael', 'Gabriel', 'Pedro', 'Mateus', 'Enzo', 'Miguel', 'Davi', 'Arthur'], 'F': ['Sofia', 'Maria', 'Isabella', 'Alice', 'Manuela', 'Laura', 'Valentina', 'Heloísa', 'Julia', 'Mariana']},
    'Japanese': {'Continent': 'Asia', 'M': ['Kenji', 'Hiroshi', 'Akira', 'Sato', 'Haruto', 'Ren', 'Kaito', 'Yuto', 'Sota', 'Riku'], 'F': ['Yumi', 'Aiko', 'Sakura', 'Hana', 'Rina', 'Yui', 'Miyu', 'Koharu', 'Aoi', 'Sana']},
    'Australian': {'Continent': 'Oceania', 'M': ['Liam', 'Noah', 'Jack', 'Ethan', 'Oliver', 'William', 'Harrison', 'Finn', 'Archie', 'Henry'], 'F': ['Mia', 'Chloe', 'Ruby', 'Isla', 'Ava', 'Matilda', 'Grace', 'Lily', 'Ella', 'Charlotte']},
    'South African': {'Continent': 'Africa', 'M': ['Sipho', 'Themba', 'Musa', 'Nkosi', 'Bheki', 'Zola', 'Khaya', 'Sifiso', 'Tshepo', 'Mandla'], 'F': ['Nomusa', 'Thandi', 'Zola', 'Khanya', 'Lindiwe', 'Nandi', 'Ayanda', 'Sizakele', 'Zanele', 'Amahle']},
    'Indian': {'Continent': 'Asia', 'M': ['Ravi', 'Arjun', 'Vikram', 'Anil', 'Siddharth', 'Rahul', 'Vivek', 'Rohan', 'Karan', 'Rajesh'], 'F': ['Priya', 'Aisha', 'Deepa', 'Lata', 'Shanti', 'Tara', 'Neha', 'Sonia', 'Kavita', 'Jaya']},
    'Nigerian': {'Continent': 'Africa', 'M': ['Chinedu', 'Olu', 'Tunde', 'Kunle', 'Ike', 'Emeka', 'Femi', 'Obi', 'Kelechi', 'Jide'], 'F': ['Aisha', 'Chiamaka', 'Ngozi', 'Yemi', 'Damilola', 'Kemi', 'Ada', 'Tolu', 'Zainab', 'Funke']},
    'Mexican': {'Continent': 'North America', 'M': ['Ricardo', 'Javier', 'Miguel', 'Alejandro', 'Luis', 'Juan', 'José', 'Carlos', 'David', 'Manuel'], 'F': ['Sofía', 'María', 'Isabella', 'Ximena', 'Camila', 'Andrea', 'Valeria', 'Natalia', 'Paulina', 'Renata']},
    'South Korean': {'Continent': 'Asia', 'M': ['Min-Jun', 'Seo-Jun', 'Do-Yun', 'Joon-Ho', 'Ji-Hoon', 'Ha-Joon', 'Hyun-Woo', 'Tae-Hyun', 'Gyu-Min', 'Sung-Min'], 'F': ['Seo-Yeon', 'Ji-Woo', 'Su-Min', 'Ha-Eun', 'Min-Ji', 'Ye-Eun', 'Yoo-Jin', 'Chae-Won', 'Ji-Min', 'Da-Eun']},
    'Spanish': {'Continent': 'Europe', 'M': ['Javier', 'Pablo', 'Daniel', 'Hugo', 'Álvaro', 'Mario', 'Adrián', 'Sergio', 'David', 'Jorge'], 'F': ['Lucía', 'María', 'Paula', 'Sara', 'Alba', 'Carla', 'Martina', 'Daniela', 'Claudia', 'Irene']},
    'Egyptian': {'Continent': 'Africa', 'M': ['Ahmed', 'Mohamed', 'Omar', 'Youssef', 'Tarek', 'Eyad', 'Karim', 'Amr', 'Ali', 'Mostafa'], 'F': ['Fatima', 'Aisha', 'Nour', 'Hana', 'Layla', 'Salma', 'Sara', 'Yasmin', 'Mariam', 'Jana']},
    'Italian': {'Continent': 'Europe', 'M': ['Francesco', 'Alessandro', 'Andrea', 'Lorenzo', 'Matteo', 'Gabriele', 'Leonardo', 'Riccardo', 'Davide', 'Simone'], 'F': ['Sofia', 'Giulia', 'Chiara', 'Aurora', 'Alice', 'Emma', 'Giorgia', 'Greta', 'Beatrice', 'Martina']},
    'Canadian': {'Continent': 'North America', 'M': ['Ethan', 'Liam', 'Noah', 'William', 'Oliver', 'Benjamin', 'Jacob', 'Lucas', 'Alexander', 'Owen'], 'F': ['Olivia', 'Emma', 'Charlotte', 'Ava', 'Sophia', 'Isabella', 'Amelia', 'Mia', 'Evelyn', 'Aria']},
    'Vietnamese': {'Continent': 'Asia', 'M': ['Bảo', 'Minh', 'Khoa', 'Huy', 'Phúc', 'Đức', 'Hoàng', 'Kiên', 'Tùng', 'Thành'], 'F': ['Ngọc', 'Mai', 'Linh', 'Trang', 'Hà', 'Phương', 'Yến', 'Thư', 'Vy', 'Quỳnh']},
    'Colombian': {'Continent': 'South America', 'M': ['Sebastián', 'Alejandro', 'Santiago', 'Nicolás', 'Daniel', 'Samuel', 'Mateo', 'Juan', 'Gabriel', 'José'], 'F': ['Salomé', 'Valeria', 'Isabella', 'Mariana', 'Camila', 'Gabriela', 'María', 'Sofía', 'Luciana', 'Daniela']},
    'Russian': {'Continent': 'Europe', 'M': ['Alexander', 'Ivan', 'Dmitry', 'Maxim', 'Sergey', 'Mikhail', 'Nikita', 'Andrei', 'Alexey', 'Vladimir'], 'F': ['Anastasia', 'Elena', 'Anna', 'Maria', 'Victoria', 'Ksenia', 'Sofia', 'Daria', 'Polina', 'Yulia']},
    'Saudi': {'Continent': 'Asia', 'M': ['Abdullah', 'Faisal', 'Khaled', 'Saud', 'Sultan', 'Bandar', 'Nawaf', 'Yazeed', 'Turki', 'Hamad'], 'F': ['Noura', 'Sara', 'Lama', 'Reema', 'Hessa', 'Joud', 'Dania', 'Fatima', 'Maha', 'Aisha']},
    'Swedish': {'Continent': 'Europe', 'M': ['Erik', 'Oscar', 'Carl', 'Gustav', 'Axel', 'Lucas', 'Elias', 'William', 'Hugo', 'Olle'], 'F': ['Emma', 'Anna', 'Sara', 'Linnéa', 'Elsa', 'Alice', 'Maja', 'Ebba', 'Wilma', 'Lovisa']},
}
PROFILE_TYPES = {
    'Culturel/Histoire': {'climate': 'Temperate', 'primary_type': 'City', 'tags': ['History', 'Cultural', 'Musées', 'Gastronomie']},
    'Aventure/Nature': {'climate': 'Temperate', 'primary_type': 'Adventure/Nature', 'tags': ['Adventure', 'Nature', 'Camping', 'Wildlife']},
    'Plage/Détente': {'climate': 'Hot', 'primary_type': 'Beach', 'tags': ['Beach', 'Tropical', 'Détente', 'Resort']},
    'Froid/Ski/Nordique': {'climate': 'Cold', 'primary_type': 'Adventure/Nature', 'tags': ['Nordic', 'Cold', 'Ski', 'Hiver-Idéal']},
    'Exotique/Budget': {'climate': 'Hot', 'primary_type': 'Cultural/Nature', 'tags': ['Budget-Friendly', 'Backpacker', 'Exotic', 'Solo-Friendly']},
    'Urbain/Luxe/Shopping': {'climate': 'Temperate', 'primary_type': 'City', 'tags': ['City', 'Luxury', 'Shopping', 'High-Cost', 'Vie-Nocturne']},
}

DESTINATIONS = [

    # 🌍 Europe (City + Culture)
    {'city': 'Rome', 'country': 'Italy', 'type': 'City', 'climate': 'Temperate', 'continent': 'Europe', 'cost_level': 'Medium', 'prestige': 4},
    {'city': 'Paris', 'country': 'France', 'type': 'City', 'climate': 'Temperate', 'continent': 'Europe', 'cost_level': 'High', 'prestige': 5},
    {'city': 'London', 'country': 'UK', 'type': 'City', 'climate': 'Temperate', 'continent': 'Europe', 'cost_level': 'High', 'prestige': 5},
    {'city': 'Barcelona', 'country': 'Spain', 'type': 'City', 'climate': 'Temperate', 'continent': 'Europe', 'cost_level': 'Medium', 'prestige': 4},
    {'city': 'Oslo', 'country': 'Norway', 'type': 'City', 'climate': 'Cold', 'continent': 'Europe', 'cost_level': 'High', 'prestige': 3},
    {'city': 'Reykjavik', 'country': 'Iceland', 'type': 'Adventure/Nature', 'climate': 'Cold', 'continent': 'Europe', 'cost_level': 'High', 'prestige': 4},
    {'city': 'Zermatt', 'country': 'Switzerland', 'type': 'Adventure/Nature', 'climate': 'Cold', 'continent': 'Europe', 'cost_level': 'High', 'prestige': 4},
    {'city': 'Budapest', 'country': 'Hungary', 'type': 'City', 'climate': 'Temperate', 'continent': 'Europe', 'cost_level': 'Low', 'prestige': 3},
    {'city': 'Prague', 'country': 'Czech Republic', 'type': 'City', 'climate': 'Temperate', 'continent': 'Europe', 'cost_level': 'Low', 'prestige': 3},
    {'city': 'Lisbon', 'country': 'Portugal', 'type': 'City', 'climate': 'Temperate', 'continent': 'Europe', 'cost_level': 'Medium', 'prestige': 3},
    {'city': 'Copenhagen', 'country': 'Denmark', 'type': 'City', 'climate': 'Cold', 'continent': 'Europe', 'cost_level': 'High', 'prestige': 4},
    {'city': 'Vienna', 'country': 'Austria', 'type': 'City', 'climate': 'Temperate', 'continent': 'Europe', 'cost_level': 'Medium', 'prestige': 4},
    {'city': 'Dublin', 'country': 'Ireland', 'type': 'City', 'climate': 'Temperate', 'continent': 'Europe', 'cost_level': 'Medium', 'prestige': 3},
    {'city': 'Amsterdam', 'country': 'Netherlands', 'type': 'City', 'climate': 'Temperate', 'continent': 'Europe', 'cost_level': 'High', 'prestige': 5},
    {'city': 'St Petersburg', 'country': 'Russia', 'type': 'City', 'climate': 'Cold', 'continent': 'Europe', 'cost_level': 'Medium', 'prestige': 3},
    {'city': 'Dubrovnik', 'country': 'Croatia', 'type': 'Beach', 'climate': 'Temperate', 'continent': 'Europe', 'cost_level': 'Medium', 'prestige': 4},
    {'city': 'Helsinki', 'country': 'Finland', 'type': 'City', 'climate': 'Cold', 'continent': 'Europe', 'cost_level': 'High', 'prestige': 3},
    {'city': 'Kraków', 'country': 'Poland', 'type': 'City', 'climate': 'Temperate', 'continent': 'Europe', 'cost_level': 'Low', 'prestige': 3},
    {'city': 'Moscow', 'country': 'Russia', 'type': 'City', 'climate': 'Cold', 'continent': 'Europe', 'cost_level': 'Medium', 'prestige': 3},
    {'city': 'Stockholm', 'country': 'Sweden', 'type': 'City', 'climate': 'Cold', 'continent': 'Europe', 'cost_level': 'High', 'prestige': 4},

    # 🏖️ Hot / Beach
    {'city': 'Malé', 'country': 'Maldives', 'type': 'Beach', 'climate': 'Hot', 'continent': 'Asia', 'cost_level': 'High', 'prestige': 5},
    {'city': 'Phuket', 'country': 'Thailand', 'type': 'Beach', 'climate': 'Hot', 'continent': 'Asia', 'cost_level': 'Low', 'prestige': 3},
    {'city': 'Cancun', 'country': 'Mexico', 'type': 'Beach', 'climate': 'Hot', 'continent': 'North America', 'cost_level': 'Medium', 'prestige': 4},
    {'city': 'Rio de Janeiro', 'country': 'Brazil', 'type': 'Beach', 'climate': 'Hot', 'continent': 'South America', 'cost_level': 'Medium', 'prestige': 4},
    {'city': 'Zanzibar', 'country': 'Tanzania', 'type': 'Beach', 'climate': 'Hot', 'continent': 'Africa', 'cost_level': 'Medium', 'prestige': 4},
    {'city': 'Bali', 'country': 'Indonesia', 'type': 'Beach', 'climate': 'Hot', 'continent': 'Asia', 'cost_level': 'Medium', 'prestige': 5},
    {'city': 'Miami', 'country': 'USA', 'type': 'Beach', 'climate': 'Hot', 'continent': 'North America', 'cost_level': 'High', 'prestige': 5},
    {'city': 'Fiji', 'country': 'Fiji', 'type': 'Beach', 'climate': 'Hot', 'continent': 'Oceania', 'cost_level': 'Medium', 'prestige': 4},
    {'city': 'Havana', 'country': 'Cuba', 'type': 'Beach', 'climate': 'Hot', 'continent': 'North America', 'cost_level': 'Low', 'prestige': 3},
    {'city': 'Cairns', 'country': 'Australia', 'type': 'Beach', 'climate': 'Hot', 'continent': 'Oceania', 'cost_level': 'Medium', 'prestige': 4},
    {'city': 'Nice', 'country': 'France', 'type': 'Beach', 'climate': 'Temperate', 'continent': 'Europe', 'cost_level': 'High', 'prestige': 4},
    {'city': 'Palawan', 'country': 'Philippines', 'type': 'Beach', 'climate': 'Hot', 'continent': 'Asia', 'cost_level': 'Medium', 'prestige': 2},
    {'city': 'Goa', 'country': 'India', 'type': 'Beach', 'climate': 'Hot', 'continent': 'Asia', 'cost_level': 'Low', 'prestige': 2},
    {'city': 'Cartagena', 'country': 'Colombia', 'type': 'Beach', 'climate': 'Hot', 'continent': 'South America', 'cost_level': 'Medium', 'prestige': 4},
    {'city': 'Perth', 'country': 'Australia', 'type': 'Beach', 'climate': 'Hot', 'continent': 'Oceania', 'cost_level': 'Medium', 'prestige': 3},

    # 🇺🇸 North America
    {'city': 'New York', 'country': 'USA', 'type': 'City', 'climate': 'Temperate', 'continent': 'North America', 'cost_level': 'High', 'prestige': 5},
    {'city': 'Banff', 'country': 'Canada', 'type': 'Adventure/Nature', 'climate': 'Cold', 'continent': 'North America', 'cost_level': 'High', 'prestige': 4},
    {'city': 'Vancouver', 'country': 'Canada', 'type': 'City', 'climate': 'Temperate', 'continent': 'North America', 'cost_level': 'High', 'prestige': 4},
    {'city': 'Los Angeles', 'country': 'USA', 'type': 'City', 'climate': 'Temperate', 'continent': 'North America', 'cost_level': 'High', 'prestige': 5},
    {'city': 'Toronto', 'country': 'Canada', 'type': 'City', 'climate': 'Temperate', 'continent': 'North America', 'cost_level': 'Medium', 'prestige': 4},
    {'city': 'Yellowstone NP', 'country': 'USA', 'type': 'Adventure/Nature', 'climate': 'Cold', 'continent': 'North America', 'cost_level': 'Medium', 'prestige': 3},
    {'city': 'Montreal', 'country': 'Canada', 'type': 'City', 'climate': 'Cold', 'continent': 'North America', 'cost_level': 'Medium', 'prestige': 4},
    {'city': 'Chicago', 'country': 'USA', 'type': 'City', 'climate': 'Temperate', 'continent': 'North America', 'cost_level': 'Medium', 'prestige': 4},
    {'city': 'San Francisco', 'country': 'USA', 'type': 'City', 'climate': 'Temperate', 'continent': 'North America', 'cost_level': 'High', 'prestige': 5},
    {'city': 'Mexico City', 'country': 'Mexico', 'type': 'City', 'climate': 'Temperate', 'continent': 'North America', 'cost_level': 'Medium', 'prestige': 4},

    # 🌏 Asia / Oceania / Africa / Others
    {'city': 'Kyoto', 'country': 'Japan', 'type': 'Cultural/Nature', 'climate': 'Temperate', 'continent': 'Asia', 'cost_level': 'Medium', 'prestige': 5},
    {'city': 'Hanoi', 'country': 'Vietnam', 'type': 'City', 'climate': 'Hot', 'continent': 'Asia', 'cost_level': 'Low', 'prestige': 2},
    {'city': 'Dubai', 'country': 'UAE', 'type': 'City', 'climate': 'Hot', 'continent': 'Asia', 'cost_level': 'High', 'prestige': 5},
    {'city': 'Seoul', 'country': 'South Korea', 'type': 'City', 'climate': 'Temperate', 'continent': 'Asia', 'cost_level': 'High', 'prestige': 5},
    {'city': 'Kathmandu', 'country': 'Nepal', 'type': 'Adventure/Nature', 'climate': 'Temperate', 'continent': 'Asia', 'cost_level': 'Low', 'prestige': 1},
    {'city': 'Cape Town', 'country': 'South Africa', 'type': 'City', 'climate': 'Temperate', 'continent': 'Africa', 'cost_level': 'Medium', 'prestige': 5},
    {'city': 'Le Caire', 'country': 'Egypt', 'type': 'Cultural/Nature', 'climate': 'Hot', 'continent': 'Africa', 'cost_level': 'Low', 'prestige': 3},
    {'city': 'Sydney', 'country': 'Australia', 'type': 'City', 'climate': 'Temperate', 'continent': 'Oceania', 'cost_level': 'High', 'prestige': 5},
    {'city': 'Queenstown', 'country': 'New Zealand', 'type': 'Adventure/Nature', 'climate': 'Temperate', 'continent': 'Oceania', 'cost_level': 'High', 'prestige': 3},
    {'city': 'Buenos Aires', 'country': 'Argentina', 'type': 'City', 'climate': 'Temperate', 'continent': 'South America', 'cost_level': 'Medium', 'prestige': 4},
    {'city': 'Cusco', 'country': 'Peru', 'type': 'Cultural/Nature', 'climate': 'Temperate', 'continent': 'South America', 'cost_level': 'Medium', 'prestige': 4},
    {'city': 'Venice', 'country': 'Italy', 'type': 'City', 'climate': 'Temperate', 'continent': 'Europe', 'cost_level': 'High', 'prestige': 5},
    {'city': 'Munich', 'country': 'Germany', 'type': 'City', 'climate': 'Temperate', 'continent': 'Europe', 'cost_level': 'Medium', 'prestige': 4},
    {'city': 'Seville', 'country': 'Spain', 'type': 'City', 'climate': 'Hot', 'continent': 'Europe', 'cost_level': 'Medium', 'prestige': 4},
    {'city': 'Edinburgh', 'country': 'UK', 'type': 'City', 'climate': 'Cold', 'continent': 'Europe', 'cost_level': 'Medium', 'prestige': 4},
    {'city': 'Geneva', 'country': 'Switzerland', 'type': 'City', 'climate': 'Cold', 'continent': 'Europe', 'cost_level': 'High', 'prestige': 5},
    {'city': 'Patagonia', 'country': 'Argentina', 'type': 'Adventure/Nature', 'climate': 'Cold', 'continent': 'South America', 'cost_level': 'Medium', 'prestige': 5},
    {'city': 'Galapagos Islands', 'country': 'Ecuador', 'type': 'Adventure/Nature', 'climate': 'Hot', 'continent': 'South America', 'cost_level': 'High', 'prestige': 5},
    {'city': 'Iguazu Falls', 'country': 'Brazil', 'type': 'Adventure/Nature', 'climate': 'Hot', 'continent': 'South America', 'cost_level': 'Low', 'prestige': 3},
    {'city': 'Marrakech', 'country': 'Morocco', 'type': 'Cultural/Nature', 'climate': 'Hot', 'continent': 'Africa', 'cost_level': 'Low', 'prestige': 3},
    {'city': 'Giza', 'country': 'Egypt', 'type': 'Cultural/Nature', 'climate': 'Hot', 'continent': 'Africa', 'cost_level': 'Low', 'prestige': 4},
    {'city': 'Kruger NP', 'country': 'South Africa', 'type': 'Adventure/Nature', 'climate': 'Hot', 'continent': 'Africa', 'cost_level': 'High', 'prestige': 5},
    {'city': 'Nairobi', 'country': 'Kenya', 'type': 'City', 'climate': 'Hot', 'continent': 'Africa', 'cost_level': 'Medium', 'prestige': 3},
    {'city': 'Bangkok', 'country': 'Thailand', 'type': 'City', 'climate': 'Hot', 'continent': 'Asia', 'cost_level': 'Low', 'prestige': 3},
    {'city': 'Shanghai', 'country': 'China', 'type': 'City', 'climate': 'Temperate', 'continent': 'Asia', 'cost_level': 'Medium', 'prestige': 5},
    {'city': 'Ubud', 'country': 'Indonesia', 'type': 'Cultural/Nature', 'climate': 'Hot', 'continent': 'Asia', 'cost_level': 'Medium', 'prestige': 4},
    {'city': 'Hoi An', 'country': 'Vietnam', 'type': 'Cultural/Nature', 'climate': 'Hot', 'continent': 'Asia', 'cost_level': 'Low', 'prestige': 3},
    {'city': 'Auckland', 'country': 'New Zealand', 'type': 'City', 'climate': 'Temperate', 'continent': 'Oceania', 'cost_level': 'Medium', 'prestige': 4},
    {'city': 'Melbourne', 'country': 'Australia', 'type': 'City', 'climate': 'Temperate', 'continent': 'Oceania', 'cost_level': 'High', 'prestige': 5},
    {'city': 'Hokkaido', 'country': 'Japan', 'type': 'Adventure/Nature', 'climate': 'Cold', 'continent': 'Asia', 'cost_level': 'Medium', 'prestige': 5},
    {'city': 'Himeji', 'country': 'Japan', 'type': 'Cultural/Nature', 'climate': 'Temperate', 'continent': 'Asia', 'cost_level': 'Medium', 'prestige': 4},
    {'city': 'Lagos', 'country': 'Nigeria', 'type': 'City', 'climate': 'Hot', 'continent': 'Africa', 'cost_level': 'Low', 'prestige': 2},
    {'city': 'Accra', 'country': 'Ghana', 'type': 'City', 'climate': 'Hot', 'continent': 'Africa', 'cost_level': 'Low', 'prestige': 2},
    {'city': 'Amman', 'country': 'Jordan', 'type': 'Cultural/Nature', 'climate': 'Hot', 'continent': 'Asia', 'cost_level': 'Medium', 'prestige': 3},
    {'city': 'Wellington', 'country': 'New Zealand', 'type': 'City', 'climate': 'Temperate', 'continent': 'Oceania', 'cost_level': 'High', 'prestige': 4},
    {'city': 'Ho Chi Minh City', 'country': 'Vietnam', 'type': 'City', 'climate': 'Hot', 'continent': 'Asia', 'cost_level': 'Low', 'prestige': 2},
    {'city': 'Bogotá', 'country': 'Colombia', 'type': 'City', 'climate': 'Temperate', 'continent': 'South America', 'cost_level': 'Medium', 'prestige': 3},
    {'city': 'Jeddah', 'country': 'Saudi Arabia', 'type': 'City', 'climate': 'Hot', 'continent': 'Asia', 'cost_level': 'High', 'prestige': 3},
    {'city': 'Medina', 'country': 'Saudi Arabia', 'type': 'Cultural/Nature', 'climate': 'Hot', 'continent': 'Asia', 'cost_level': 'High', 'prestige': 3},
    {'city': 'Pékin', 'country': 'China', 'type': 'City', 'climate': 'Temperate', 'continent': 'Asia', 'cost_level': 'Medium', 'prestige': 4},
    {'city': 'Delhi', 'country': 'India', 'type': 'City', 'climate': 'Hot', 'continent': 'Asia', 'cost_level': 'Low', 'prestige': 2},
    {'city': 'Caracas', 'country': 'Venezuela', 'type': 'City', 'climate': 'Hot', 'continent': 'South America', 'cost_level': 'Low', 'prestige': 1},
    {'city': 'San Juan', 'country': 'Puerto Rico', 'type': 'Beach', 'climate': 'Hot', 'continent': 'North America', 'cost_level': 'Medium', 'prestige': 3},
    {'city': 'Boston', 'country': 'USA', 'type': 'City', 'climate': 'Temperate', 'continent': 'North America', 'cost_level': 'High', 'prestige': 4},
    {'city': 'San Diego', 'country': 'USA', 'type': 'City', 'climate': 'Temperate', 'continent': 'North America', 'cost_level': 'High', 'prestige': 4},
    {'city': 'Kuala Lumpur', 'country': 'Malaysia', 'type': 'City', 'climate': 'Hot', 'continent': 'Asia', 'cost_level': 'Low', 'prestige': 3},
    {'city': 'Chiang Mai', 'country': 'Thailand', 'type': 'Cultural/Nature', 'climate': 'Hot', 'continent': 'Asia', 'cost_level': 'Low', 'prestige': 3},

]

DURATION_MAPPING = {
    'Beach': (5, 12),
    'City': (3, 7),
    'Cultural/Nature': (7, 15),
    'Adventure/Nature': (7, 15)
}
ACCOMMODATION_COHERENCE = {
    'City': ['Hotel', 'Airbnb', 'Guesthouse', 'Hostel'],
    'Beach': ['Resort', 'Hotel', 'Villa', 'Airbnb', 'Guesthouse'],
    'Cultural/Nature': ['Guesthouse', 'Airbnb', 'Hostel', 'Hotel'],
    'Adventure/Nature': ['Camping', 'Hostel', 'Guesthouse', 'Airbnb', 'Hotel'],
}
NON_CITY_CAMPING = ['Reykjavik', 'Zermatt', 'Yellowstone NP', 'Banff', 'Patagonia', 'Galapagos Islands', 'Iguazu Falls', 'Kruger NP', 'Hokkaido', 'Kathmandu']

# Ajout : liste générique de modes de transport possibles pour préférence utilisateur
TRANSPORT_MODES = [
    'Bus', 'Métro', 'Marche', 'Taxi', 'Vélo', 'Tuk-Tuk', 'Scooter', 'Voiture', 'Voiture (si isolé)', '4x4', 'Randonnée', 'Bateau', 'Transports Publics'
]

# --- NOUVELLES FONCTIONS DE COHÉRENCE ---

def get_base_accommodation_cost(acc_type, cost_level):
    """Coût JOURNALIER d'hébergement"""
    cost_matrix = {
        'Hostel': {'Low': 20, 'Medium': 35, 'High': 55},
        'Camping': {'Low': 10, 'Medium': 20, 'High': 35},
        'Airbnb': {'Low': 50, 'Medium': 85, 'High': 140},
        'Hotel': {'Low': 70, 'Medium': 120, 'High': 200},
        'Resort': {'Low': 150, 'Medium': 280, 'High': 450},
        'Villa': {'Low': 120, 'Medium': 220, 'High': 350},
        'Ryokan': {'Low': 80, 'Medium': 140, 'High': 240},
        'Guesthouse': {'Low': 30, 'Medium': 55, 'High': 90},
    }
    return cost_matrix[acc_type][cost_level] * np.random.uniform(0.85, 1.15)

def get_average_daily_cost(cost_level):
    """Coût journalier HORS hébergement"""
    base_cost = DAILY_COST_MAPPING.get(cost_level, 50.00)
    return base_cost * np.random.uniform(0.9, 1.1)

def get_duration(dest_type):
    min_d, max_d = DURATION_MAPPING.get(dest_type, (5, 10))
    return np.random.randint(min_d, max_d + 1)

def generate_users():
    users = []
    user_transport_profiles = {}

    for i in range(1, NB_USERS + 1):
        user_id = f"U{i:04d}"
        nationality = random.choice(list(COHERENCE_MAPPING.keys()))
        gender = random.choice(['Male', 'Female'])
        
        mapping = COHERENCE_MAPPING[nationality]
        name = random.choice(mapping['M' if gender == 'Male' else 'F'])
        profile_type = random.choice(list(PROFILE_TYPES.keys()))
        
        profile_data = PROFILE_TYPES[profile_type]
        
        core_modes = random.sample(TRANSPORT_MODES, k=3)
        weights = np.random.dirichlet(alpha=[3,2,1])
        transport_profile = {m: w for m, w in zip(core_modes, weights)}
        user_transport_profiles[user_id] = transport_profile
        
        users.append({
            'User ID': user_id,
            'Traveler name': name,
            'Traveler age': np.random.randint(18, 70),
            'Traveler gender': gender,
            'Traveler nationality': nationality,
            'Profile Type': profile_type,
            'Climate Pref': profile_data['climate'],
            'Primary Dest Type': profile_data['primary_type'],
            'Acc Pref': random.choice(['Hotel', 'Hostel', 'Airbnb', 'Villa', 'Guesthouse']),
            'Transport Core Modes': ';'.join(core_modes),
            'Traveler Continent': mapping['Continent']
        })
    
    return pd.DataFrame(users), user_transport_profiles

def generate_trips(df_users, user_transport_profiles):
    all_trips = []
    trip_counter = 1
    
    # FIX 1: Autoriser les répétitions de pays
    history = {uid: {'dates': []} for uid in df_users['User ID']}
    
    for _, user_row in df_users.iterrows():
        uid = user_row['User ID']
        profile = user_row
        transport_profile = user_transport_profiles[uid]
        
        nb_trips = int(np.clip(np.random.normal(6, 2.5), 1, 20))
        
        for _ in range(nb_trips):
            # FIX 2: Pondération par popularité + prestige (pas seulement prestige)
            # Accepter des variations (pas de filtre strict)
            climate_match = [d for d in DESTINATIONS if d['climate'] == profile['Climate Pref']]
            type_match = [d for d in DESTINATIONS if d['type'] == profile['Primary Dest Type']]
            
            # 60% de chance de respecter les préférences, 40% d'explorer
            if random.random() < 0.6 and climate_match:
                candidates = climate_match
            elif random.random() < 0.6 and type_match:
                candidates = type_match
            else:
                candidates = DESTINATIONS
            
            # FIX 3: Pondération réaliste (popularité > prestige)
            weights = np.array([
                d.get('popularity', 3) * (d.get('prestige', 3)  ** 2.5)
                for d in candidates
            ], dtype=float)
            weights /= weights.sum()
            
            destination = np.random.choice(candidates, p=weights)
            dest_city = f"{destination['city']}, {destination['country']}"
            
            # Dates sans overlap
            duration = get_duration(destination['type'])
            max_attempts = 50
            for attempt in range(max_attempts):
                y = np.random.randint(START_YEAR, END_YEAR+1)
                m = np.random.randint(1, 13)
                d = np.random.randint(1, 28)
                start_d = date(y, m, d)
                end_d = start_d + timedelta(days=duration-1)
                
                overlap = any(start_d <= e_old and end_d >= s_old 
                             for s_old, e_old in history[uid]['dates'])
                
                if not overlap:
                    history[uid]['dates'].append((start_d, end_d))
                    break
            
            # Hébergement cohérent
            acc_options = ACCOMMODATION_COHERENCE.get(destination['type'], ['Hotel'])
            if profile['Acc Pref'] in acc_options and random.random() < 0.7:
                acc_type = profile['Acc Pref']
            else:
                acc_type = random.choice(acc_options)
            
            # FIX 4: Coût total = hébergement + dépenses quotidiennes
            acc_cost_per_day = get_base_accommodation_cost(acc_type, destination['cost_level'])
            daily_expenses = get_average_daily_cost(destination['cost_level'])
            
            total_accommodation_cost = round(acc_cost_per_day * duration, 2)
            total_daily_expenses = round(daily_expenses * duration, 2)
            
            # Transport local
            p = np.array([transport_profile.get(m, 0.1) for m in TRANSPORT_MODES], dtype=float)
            p = p / p.sum()
            chosen_transport = np.random.choice(TRANSPORT_MODES, p=p)
            
            all_trips.append({
                'Trip ID': f"T{trip_counter:05d}",
                'User ID': uid,
                'Destination': dest_city,
                'Start date': start_d.isoformat(),
                'End date': end_d.isoformat(),
                'Duration (days)': duration,
                'Traveler name': profile['Traveler name'],
                'Traveler age': profile['Traveler age'],
                'Traveler gender': profile['Traveler gender'],
                'Traveler nationality': profile['Traveler nationality'],
                'Accommodation type': acc_type,
                'Accommodation cost': total_accommodation_cost,
                'Average Daily Cost': total_daily_expenses,
                'Total cost': round(total_accommodation_cost + total_daily_expenses, 2),
                'Local Transport Mode': chosen_transport,
            })
            
            trip_counter += 1
    
    return pd.DataFrame(all_trips)

def main():
    print("Génération des utilisateurs...")
    df_users, user_transport_profiles = generate_users()
    
    print("Génération des voyages...")
    df_trips = generate_trips(df_users, user_transport_profiles)
    
    print("Export CSV...")
    df_users.to_csv("users_generated.csv", index=False)
    df_trips.to_csv("travel_generated.csv", index=False)
    
    print("Terminé !")
    print(f"{len(df_users)} utilisateurs générés")
    print(f"{len(df_trips)} voyages générés")
    
    # Stats de vérification
    print("\n--- Top 10 destinations ---")
    print(df_trips['Destination'].value_counts().head(10))
    
    print("\n--- Coût moyen par type d'hébergement ---")
    print(df_trips.groupby('Accommodation type')['Accommodation cost'].mean().sort_values(ascending=False))

if __name__ == "__main__":
    main()