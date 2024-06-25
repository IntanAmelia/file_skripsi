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
menu = st.sidebar.radio("Go to", ["Imputasi Missing Value Menggunakan KNN", "Deteksi Outlier", "Normalisasi Data", "Model LSTM", "Prediksi LSTM", "Implementasi"])

# Add different sections based on the selected menu item
if menu == "Imputasi Missing Value Menggunakan KNN":
    st.header("Welcome to the Home Page")
    st.write("This is the main area of the Home page.")
elif menu == "Deteksi Outlier":
    st.header("About Us")
    st.write("This section contains information about us.")
elif menu == "Normalisasi Data":
    st.header("Settings")
    st.write("Configure your preferences here.")
elif menu == "Model LSTM":
    st.header("Contact Us")
    st.write("Get in touch with us here.")
elif menu == "Prediksi LSTM":
    st.header("Settings")
    st.write("Configure your preferences here.")
elif menu == "Implementasi":
    st.header("Contact Us")
    st.write("Get in touch with us here.")
    
# Display a message in the main area
st.write("This is the main area of the app.")
