# app.py
#import library

import pickle
import streamlit as st
import pandas as pd
import numpy as np
import keras
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import math
import joblib
import tensorflow as tf
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.models import load_model

# Set the title of the app
st.title("PREDIKSI CURAH HUJAN MENGGUNAKAN LSTM DAN K-NN DALAM IMPUTASI MISSING VALUE")

# Add a sidebar title
st.sidebar.title("Main Menu")

# Add a sidebar header and multiple menu items
st.sidebar.header("Navigation")
menu = st.sidebar.radio("Go to", ["Home", "About", "Settings", "Contact"])

# Add different sections based on the selected menu item
if menu == "Home":
    st.header("Welcome to the Home Page")
    st.write("This is the main area of the Home page.")
elif menu == "About":
    st.header("About Us")
    st.write("This section contains information about us.")
elif menu == "Settings":
    st.header("Settings")
    st.write("Configure your preferences here.")
elif menu == "Contact":
    st.header("Contact Us")
    st.write("Get in touch with us here.")

# Add more widgets to the sidebar
st.sidebar.header("User Information")
user_name = st.sidebar.text_input("Enter your name:")
user_age = st.sidebar.number_input("Enter your age:", min_value=1, max_value=120, value=25)
user_gender = st.sidebar.radio("Gender", ["Male", "Female", "Other"])

# Add more options
st.sidebar.header("Preferences")
favorite_color = st.sidebar.selectbox("Choose your favorite color:", ["Red", "Green", "Blue", "Yellow", "Other"])
hobbies = st.sidebar.multiselect("Select your hobbies:", ["Reading", "Traveling", "Gaming", "Cooking", "Other"])
volume = st.sidebar.slider("Set the volume level", 0, 100, 50)

# Add actions
st.sidebar.header("Actions")
if st.sidebar.button("Submit"):
    st.sidebar.write("Form Submitted!")
    st.write(f"Hello, {user_name}!")
    st.write(f"Your age is {user_age}.")
    st.write(f"Your gender is {user_gender}.")
    st.write(f"Your favorite color is {favorite_color}.")
    st.write(f"Your hobbies are {', '.join(hobbies)}.")
    st.write(f"The volume level is set to {volume}.")

# Display a message in the main area
st.write("This is the main area of the app.")
