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
        
        model_knn = st.radio("Pemodelan", ('Imputasi Missing Value', 'Normalisasi Data', 'Prediksi Menggunakan LSTM', 'Grafik Perbandingan Data Asli dengan Hasil Prediksi'))
        if model_knn == 'Imputasi Missing Value':
            st.write('Dataset yang telah Dilakukan Proses Imputasi Missing Value :')
            df_imputed = pd.read_csv('imputasi_n_3.csv')
            st.write(df_imputed)

        elif model_knn == 'Normalisasi Data':
            st.write('Data yang telah Dilakukan Proses Normalisasi Data :')
            df_normalisasi = pd.read_csv('normalisasi_n_3.csv')
            st.write(df_normalisasi)

        elif model_knn == 'Prediksi Menggunakan LSTM':
            # Menampilkan hasil prediksi
            st.write("Hasil Prediksi:")
            df_prediksi = pd.read_csv('predictions_knn_n_3_epochs_12_lr_0.01_ts_50.csv')
            st.write(df_prediksi)

            # Menampilkan RMSE
            y_test = pd.read_csv('ytest_knn_n_3_epochs_12_lr_0.01_ts_50.csv')
            rmse = np.sqrt(np.mean((df_prediksi.values - y_test.values)**2))
            st.write('RMSE : ')
            st.write(rmse)

        elif model_knn == 'Grafik Perbandingan Data Asli dengan Hasil Prediksi':
            df_imputed = pd.read_csv('imputasi_n_3.csv')
            df_imputed['Tanggal'] = pd.to_datetime(df_imputed['Tanggal'])
            df_normalisasi = pd.read_csv('normalisasi_n_3.csv')
            df_prediksi = pd.read_csv('predictions_knn_n_3_epochs_12_lr_0.01_ts_50.csv')
            
            plt.figure(figsize=(20, 7))
            plt.plot(df_imputed['Tanggal'][1193:], df_imputed['RR'][1193:], color='blue', label='Curah Hujan Asli')
            plt.plot(df_imputed['Tanggal'][1193:], df_normalisasi['normalisasi'][1193:], color='green', label='Normalisasi')
            plt.plot(df_imputed['Tanggal'][1193:], df_prediksi['prediksi'], color='red', label='Prediksi Curah Hujan')
            plt.title('Prediksi Curah Hujan')
            plt.xlabel('Tanggal')
            plt.ylabel('Curah Hujan (mm)')
            plt.legend()
            # Menampilkan plot di Streamlit
            st.pyplot(plt)
         
    with tab3:
        st.write("""
        <h5>Menghapus Data yang Terdapat Missing Value</h5>
        <br>
        """, unsafe_allow_html=True)
        
        model_hapusdata = st.radio("Pemodelan", ('Hapus Data yang Terdapat Missing Value', 'Normalisasi Data', 'Prediksi Menggunakan LSTM', 'Grafik Perbandingan Data Asli dengan Hasil Prediksi'))
        if model_hapusdata == 'Hapus Data yang Terdapat Missing Value':
            st.write('Dataset yang telah Dilakukan Proses Imputasi Missing Value :')
            df_imputed = pd.read_csv('hapus_data.csv')
            st.write(df_imputed)

        elif model_hapusdata == 'Normalisasi Data':
            st.write('Data yang telah Dilakukan Proses Normalisasi Data :')
            df_normalisasi = pd.read_csv('normalisasi.csv')
            st.write(df_normalisasi)

        elif model_hapusdata == 'Prediksi Menggunakan LSTM': 
            # Menampilkan hasil prediksi
            st.write("Hasil Prediksi:")
            prediksi = pd.read_csv('predictions_hapusdata_epochs_12_lr_0.001_ts_75.csv')
            st.write(prediksi)

            # Menampilkan RMSE
            y_test = pd.read_csv('ytest_hapusdata_epochs_12_lr_0.001_ts_75.csv')
            rmse = np.sqrt(np.mean((prediksi.values - y_test.values)**2))
            st.write('RMSE : ')
            st.write(rmse)

        elif model_hapusdata == 'Grafik Perbandingan Data Asli dengan Hasil Prediksi':
            df_imputed = pd.read_csv('hapus_data.csv')
            df_imputed['Tanggal'] = pd.to_datetime(df_imputed['Tanggal'])
            df_normalisasi = pd.read_csv('normalisasi.csv')
            prediksi = pd.read_csv('predictions_hapusdata_epochs_12_lr_0.001_ts_75.csv')
            
            plt.figure(figsize=(20, 7))
            plt.plot(df_imputed['Tanggal'][999:], df_imputed['RR'][999:], color='blue', label='Curah Hujan Asli')
            plt.plot(df_imputed['Tanggal'][999:], df_normalisasi['0'][999:], color='green', label='Normalisasi')
            plt.plot(df_imputed['Tanggal'][999:], prediksi['0'], color='red', label='Prediksi Curah Hujan')
            plt.title('Prediksi Curah Hujan')
            plt.xlabel('Tanggal')
            plt.ylabel('Curah Hujan (mm)')
            plt.legend()
            # Menampilkan plot di Streamlit
            st.pyplot(plt)
        
    with tab4:
        n = 2  # Example: Predict the next 10 time steps
        future_predictions = []
        x_test = pd.read_csv('xtest_knn_n_3_epochs_12_lr_0.01_ts_50.csv')
        df_imputed = pd.read_csv('imputasi_n_3.csv')
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df_imputed[['RR']])
        scaled_data_df = pd.DataFrame(scaled_data)
        values = scaled_data_df.values
        df_imputed['Tanggal'] = pd.to_datetime(df_imputed['Tanggal'])
        model_path = 'model_knn_n_3_epochs_12_lr_0.01_ts_50.h5'
        model = tf.keras.models.load_model(model_path)
        model_path_pathlib = 'model_knn_n_3_epochs_12_lr_0.01_ts_50.h5'
        model = tf.keras.models.load_model(model_path_pathlib)
        x_last_window = x_test.iloc[-1].values.reshape((1, -1, 1))
        
        for _ in range(n):
            # Predict the next time step
            prediction = model.predict(x_last_window)
        
            # Append the prediction to the list of future predictions
            future_predictions.append(prediction[0])
        
            # Update the last window by removing the first element and appending the prediction
            x_last_window = np.append(x_last_window[:, 1:, :], prediction.reshape(1, 1, 1), axis=1)
        
        # Convert the list of future predictions to a numpy array
        future_predictions = np.array(future_predictions)
        
        # Inverse transform predictions to get the original scale
        future_predictions_denormalisasi = scaler.inverse_transform(future_predictions)
        st.write('Prediksi Selanjutnya : ')
        st.write(future_predictions_denormalisasi)

        # Plotting the predictions
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(future_predictions_denormalisasi)), future_predictions_denormalisasi, marker='o', color='red', label='Future Predictions')
        plt.title('Prediksi Curah Hujan Selanjutnya')
        plt.xlabel('Time Step')
        plt.ylabel('Curah Hujan (mm)')
        plt.legend()
        st.pyplot(plt)
        
if __name__ == "__main__":
    main()
