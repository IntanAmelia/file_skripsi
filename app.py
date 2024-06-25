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
menu = st.sidebar.radio("Go to", ["Dataset", "Imputasi Missing Value Menggunakan KNN", "Deteksi Outlier", "Normalisasi Data", "Model LSTM", "Prediksi LSTM", "Implementasi"])

if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_imputed' not in st.session_state:
    st.session_state.df_imputed = None
# Add different sections based on the selected menu item
if menu == "Dataset":
    st.write("""
    <h5>Data Understanding</h5>
    <br>
    """, unsafe_allow_html=True)

    st.markdown("""
    Link Dataset:
    https://dataonline.bmkg.go.id
    """, unsafe_allow_html=True)


    st.write('Dataset ini berisi tentang curah hujan')
    missing_values = ['8888']
    df = pd.read_excel('Dataset_Curah_Hujan.xlsx', na_values = missing_values)
    st.session_state.df = df
    st.write("Dataset Curah Hujan : ")
    st.write(df)
elif menu == "Imputasi Missing Value Menggunakan KNN":
    df = st.session_state.df
    if df is not None:
        missing_data = df[df.isna().any(axis=1)]
        st.write('Data yang Mempunyai Missing Value :')
        st.write(missing_data)
        k = st.selectbox("Pilih nilai k (jumlah tetangga terdekat) :", [3, 4, 5])
        preprocessing = KNNImputer(n_neighbors=k)
        data_imputed = preprocessing.fit_transform(df[['RR']])
        df_imputed = df.copy()
        df_imputed['RR_Imputed'] = data_imputed
        st.session_state.df_imputed = df_imputed
        df_comparison = df_imputed[['Tanggal', 'RR', 'RR_Imputed']]
        st.write('Data yang telah dilakukan Proses Imputasi Missing Value dengan KNN')
        st.write(df_comparison)
    else:
        st.write("Silahkan masukkan dataset terlebih dahulu.")
elif menu == "Deteksi Outlier":
    df_imputed = st.session_state.df_imputed
    if df_imputed is not None:
        mean_rainfall = df_imputed['RR'].mean()
        std_rainfall = df_imputed['RR'].std()
        threshold = 3 * std_rainfall
        outliers = np.abs(df_imputed['RR'] - mean_rainfall) > threshold
        df_imputed['Outlier'] = outliers
        st.session_state.df_imputed = df_imputed
        st.write('Dataset yang termasuk outlier :')
        st.dataframe(df_imputed.style.format({'Outlier': '{0}'}))
    else:
        st.write('Silahkan melakukan imputasi missing value terlebih dahulu.')
elif menu == "Normalisasi Data":
    df_imputed = st.session_state.df_imputed
    if df_imputed is not None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df_imputed[['RR']])
        df_normalisasi = pd.DataFrame(scaled_data)
        st.session_state.scaled_data = scaled_data
        st.write('Data setelah dilakukan normalisasi :')
        st.write(df_normalisasi)
    else:
        st.write('Silahkan masukkan dataset terlebih dahulu')
elif menu == "Model LSTM":
    st.header("Contact Us")
    st.write("Get in touch with us here.")
elif menu == "Prediksi LSTM":
    st.header("Settings")
    st.write("Configure your preferences here.")
elif menu == "Implementasi":
    st.header("Contact Us")
    st.write("Get in touch with us here.")
    
