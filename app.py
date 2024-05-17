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

def main():
    st.set_page_config(
    page_title="PREDIKSI CURAH HUJAN MENGGUNAKAN LSTM DAN K-NN DALAM IMPUTASI MISSING VALUE"
)
    st.title('PREDIKSI CURAH HUJAN MENGGUNAKAN LSTM DAN K-NN DALAM IMPUTASI MISSING VALUE')
    
    tab1, tab2, tab3, tab4 = st.tabs(["Data Understanding", "Imputasi Missing Value Menggunakan KNN", "Hapus Data yang terdapat Missing Value", "Prediksi Selanjutnya"])

    with tab1:
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
        st.write("Dataset Curah Hujan : ")
        st.write(df)
        
    with tab2:
        st.write("""
        <h5>Imputasi Missing Value Menggunakan KNN</h5>
        <br>
        """, unsafe_allow_html=True)
        st.write("""
        Pada skenario ini akan dibagi menjadi beberapa parameter, yakni sebagai berikut : 
        <ol>
        <li> K = 3; batch size = 32; hidden layer = 100; learning rate = 0.01; epoch = 12; time step = 25 </li>
        <li> K = 4; batch size = 32; hidden layer = 100; learning rate = 0.001; epoch = 25; time step = 50 </li>
        <li> K = 5; batch size = 32; hidden layer = 100; learning rate = 0.0001; epoch = 50; time step = 75 </li>
        </ol>
        """,unsafe_allow_html=True)

        preprocessing = st.radio(
        "Preprocessing Data",
        ('K = 3; batch size = 32; hidden layer = 100; learning rate = 0.01; epoch = 12; time step = 25',
         'K = 4; batch size = 32; hidden layer = 100; learning rate = 0.001; epoch = 25; time step = 50',
         'K = 5; batch size = 32; hidden layer = 100; learning rate = 0.0001; epoch = 50; time step = 75'))
        if preprocessing == 'K = 3; batch size = 32; hidden layer = 100; learning rate = 0.01; epoch = 12; time step = 25':
            model_path = 'model_lstm_knn_s1.hdf5'
            model = tf.keras.models.load_model(model_path)
            model_path_pathlib = 'model_lstm_knn_s1.hdf5'
            model = tf.keras.models.load_model(model_path_pathlib)
            
            # Memuat data testing (x_test)
            x_test = pd.read_csv('x_test_knn_s1.csv')
            
            # Melakukan prediksi
            predictions = model.predict(x_test)
             
            # Menampilkan hasil prediksi
            st.write("Hasil Prediksi:")
            st.write(predictions)
            
        elif preprocessing == 'K = 4; batch size = 32; hidden layer = 100; learning rate = 0.001; epoch = 25; time step = 50':
            model_path = 'model_lstm_knn_s2.hdf5'
            model = tf.keras.models.load_model(model_path)
            model_path_pathlib = 'model_lstm_knn_s2.hdf5'
            model = tf.keras.models.load_model(model_path_pathlib)
            
            # Memuat data testing (x_test)
            x_test = pd.read_csv('x_test_knn_s2.csv')
            
            # Melakukan prediksi
            predictions = model.predict(x_test)
            
            # Menampilkan hasil prediksi
            st.write("Hasil Prediksi:")
            st.write(predictions)
            
        elif preprocessing == 'K = 5; batch size = 32; hidden layer = 100; learning rate = 0.0001; epoch = 50; time step = 75':
            model_path = 'model_lstm_knn_s3.hdf5'
            model = tf.keras.models.load_model(model_path)
            model_path_pathlib = 'model_lstm_knn_s3.hdf5'
            model = tf.keras.models.load_model(model_path_pathlib)
            
            # Memuat data testing (x_test)
            x_test = pd.read_csv('x_test_knn_s3.csv')
            
            # Melakukan prediksi
            predictions = model.predict(x_test)
            
            # Menampilkan hasil prediksi
            st.write("Hasil Prediksi:")
            st.write(predictions)

    with tab3:
        st.write("""
        <h5>Menghapus Data yang Terdapat Missing Value</h5>
        <br>
        """, unsafe_allow_html=True)
        st.write("""
        Pada skenario ini akan dibagi menjadi beberapa parameter, yakni sebagai berikut : 
        <ol>
        <li> Batch size = 32; hidden layer = 100; learning rate = 0.01; epoch = 12; time step = 25 </li>
        <li> Batch size = 32; hidden layer = 100; learning rate = 0.001; epoch = 25; time step = 50 </li>
        <li> Batch size = 32; hidden layer = 100; learning rate = 0.0001; epoch = 50; time step = 75 </li>
        </ol>
        """,unsafe_allow_html=True)

        preprocessing = st.radio(
        "Preprocessing Data",
        ('Batch size = 32; hidden layer = 100; learning rate = 0.01; epoch = 12; time step = 25',
         'Batch size = 32; hidden layer = 100; learning rate = 0.001; epoch = 25; time step = 50',
         'Batch size = 32; hidden layer = 100; learning rate = 0.0001; epoch = 50; time step = 75'))
        if preprocessing == 'Batch size = 32; hidden layer = 100; learning rate = 0.01; epoch = 12; time step = 25':
            model_path = 'model_lstm_hapusdata_s1.hdf5'
            model = tf.keras.models.load_model(model_path)
            model_path_pathlib = 'model_lstm_hapusdata_s1.hdf5'
            model = tf.keras.models.load_model(model_path_pathlib)
            
            # Memuat data testing (x_test)
            x_test = pd.read_csv('x_test_hapusdata_s1.csv')
            
            # Melakukan prediksi
            predictions = model.predict(x_test)
            
            # Menampilkan hasil prediksi
            st.write("Hasil Prediksi:")
            st.write(predictions)
            
        elif preprocessing == 'Batch size = 32; hidden layer = 100; learning rate = 0.001; epoch = 25; time step = 50':
            model_path = 'model_lstm_hapusdata_s2.hdf5'
            model = tf.keras.models.load_model(model_path)
            model_path_pathlib = 'model_lstm_hapusdata_s2.hdf5'
            model = tf.keras.models.load_model(model_path_pathlib)
            
            # Memuat data testing (x_test)
            x_test = pd.read_csv('x_test_hapusdata_s2.csv')
            
            # Melakukan prediksi
            predictions = model.predict(x_test)
            
            # Menampilkan hasil prediksi
            st.write("Hasil Prediksi:")
            st.write(predictions)
            
        elif preprocessing == 'Batch size = 32; hidden layer = 100; learning rate = 0.0001; epoch = 50; time step = 75':
            model_path = 'model_lstm_hapusdata_s3.hdf5'
            model = tf.keras.models.load_model(model_path)
            model_path_pathlib = 'model_lstm_hapusdata_s3.hdf5'
            model = tf.keras.models.load_model(model_path_pathlib)
            
            # Memuat data testing (x_test)
            x_test = pd.read_csv('x_test_hapusdata_s3.csv')
            
            # Melakukan prediksi
            predictions = model.predict(x_test)
            
            # Menampilkan hasil prediksi
            st.write("Hasil Prediksi:")
            st.write(predictions)

    # with tab4:

    
        
if __name__ == "__main__":
    main()
