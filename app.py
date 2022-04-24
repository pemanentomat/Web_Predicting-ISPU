from pyexpat import model
import MySQLdb
from flask import Flask, jsonify, render_template, url_for, redirect, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import pyplot
from datetime import datetime, timedelta
from math import sqrt
from numpy import concatenate
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
import tensorflow as tf
import h5py
from tensorflow.keras.models import load_model
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import h5py
import csv
from sqlalchemy import create_engine, types
from flask_sqlalchemy import SQLAlchemy
from flask_mysqldb import MySQL
from flask_migrate import Migrate

app = Flask(__name__)
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'dataset_ispu'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'
mysql = MySQL(app)

@app.route("/")
def home():
    return render_template('index.html')


@app.route("/Dataset")
def Dataset():
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cur.execute("SELECT * FROM datasetbersih ORDER BY id ASC")
    dtdataset = cur.fetchall()
    cur.close()
    return render_template('Dataset.html', data=dtdataset)

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    stasiun_selected = ''
    zat_selected = ''

    if request.method == 'GET':
        stasiun_selected = 'DKI1 (Bunderan HI)'
        zat_selected = 'pm10'
    elif request.method == 'POST':
        stasiun_selected = request.form.get('stasiun')
        zat_selected = request.form.get('zat')

    train_size = {
        'pm10': 0.8,
        'so2': 0.5,
        'co': 0.7,
        'o3': 0.8,
        'no2': 0.6
    }

    set_param_SVR = {
        'pm10': {
            'DKI1 (Bunderan HI)': {
                'Save_model': tf.keras.models.load_model('save_model\model_pm10.h5')
            },
            'DKI2 (Kelapa Gading)': {
                'Save_model': tf.keras.models.load_model('save_model\model_pm10.h5')
            },
            'DKI3 (Jagakarsa)': {
                'Save_model': tf.keras.models.load_model('save_model\model_pm10.h5')
            },
            "DKI4 (Lubang Buaya)": {
                'Save_model': tf.keras.models.load_model('save_model\model_pm10.h5')
            },
            "DKI5 (Kebon Jeruk)": {
                'Save_model': tf.keras.models.load_model('save_model\model_pm10.h5')
            }
        },
        'so2': {
            'DKI1 (Bunderan HI)': {
                'Save_model': tf.keras.models.load_model('save_model\model_no2.h5')
            },
            'DKI2 (Kelapa Gading)': {
                'Save_model': tf.keras.models.load_model('save_model\model_no2.h5')
            },
            'DKI3 (Jagakarsa)': {
                'Save_model': tf.keras.models.load_model('save_model\model_no2.h5')
            },
            "DKI4 (Lubang Buaya)": {
                'Save_model': tf.keras.models.load_model('save_model\model_no2.h5')
            },
            "DKI5 (Kebon Jeruk)": {
                'Save_model': tf.keras.models.load_model('save_model\model_no2.h5')
            }
        },
        'co': {
            'DKI1 (Bunderan HI)': {
                'Save_model': tf.keras.models.load_model('save_model\model_co.h5')

            },
            'DKI2 (Kelapa Gading)': {
                'Save_model': tf.keras.models.load_model('save_model\model_co.h5')

            },
            'DKI3 (Jagakarsa)': {
                'Save_model': tf.keras.models.load_model('save_model\model_co.h5')

            },
            "DKI4 (Lubang Buaya)": {
                'Save_model': tf.keras.models.load_model('save_model\model_co.h5')

            },
            "DKI5 (Kebon Jeruk)": {
                'Save_model': tf.keras.models.load_model('save_model\model_co.h5')

            }
        },
        'o3': {
            'DKI1 (Bunderan HI)': {
                'Save_model': tf.keras.models.load_model('save_model\model_o3.h5')

            },
            'DKI2 (Kelapa Gading)': {
                'Save_model': tf.keras.models.load_model('save_model\model_o3.h5')

            },
            'DKI3 (Jagakarsa)': {
                'Save_model': tf.keras.models.load_model('save_model\model_o3.h5')

            },
            "DKI4 (Lubang Buaya)": {
                'Save_model': tf.keras.models.load_model('save_model\model_o3.h5')

            },
            "DKI5 (Kebon Jeruk)": {
                'Save_model': tf.keras.models.load_model('save_model\model_o3.h5')

            }
        },
        'no2': {
            'DKI1 (Bunderan HI)': {
                'Save_model': tf.keras.models.load_model('save_model\model_no2.h5')

            },
            'DKI2 (Kelapa Gading)': {
                'Save_model': tf.keras.models.load_model('save_model\model_no2.h5')

            },
            'DKI3 (Jagakarsa)': {
                'Save_model': tf.keras.models.load_model('save_model\model_no2.h5')

            },
            "DKI4 (Lubang Buaya)": {
                'Save_model': tf.keras.models.load_model('save_model\model_no2.h5')

            },
            "DKI5 (Kebon Jeruk)": {
                'Save_model': tf.keras.models.load_model('save_model\model_no2.h5')

            }
        }

    }

    df_new = pd.read_csv(r'kombinasisekian.csv', parse_dates=["tanggal"])

    df_new['tanggal'] = pd.to_datetime(df_new['tanggal'])
    df_new['tanggal'] = df_new['tanggal'].dt.strftime('%Y-%m-%d')

    df_new = df_new.replace('---', np.nan)

    df_new.dropna(subset=["tanggal"], inplace=True)

    df_new['stasiun'] = df_new['stasiun'].replace(
        'DKI5 (Kebon Jeruk) Jakarta Barat', 'DKI5 (Kebon Jeruk)')

    df_new['pm10'] = df_new['pm10'].astype(str).astype(float)
    df_new['so2'] = df_new['so2'].astype(str).astype(float)
    df_new['co'] = df_new['co'].astype(str).astype(float)
    df_new['o3'] = df_new['o3'].astype(str).astype(float)
    df_new['no2'] = df_new['no2'].astype(str).astype(float)

    df_new = df_new.interpolate(method='linear', axis=0)

    df_stasiun = df_new[(df_new["stasiun"] == stasiun_selected)]
    df_stasiun = df_stasiun.sort_values(by="tanggal")
    df_stasiun = df_stasiun.reset_index(drop=True)

    df_stasiun["tanggal"] = pd.to_datetime(df_stasiun["tanggal"])
    series_zat = df_stasiun.copy()
    series_zat = df_stasiun[['tanggal', zat_selected]]

    trainpm10_dates = pd.to_datetime(series_zat['tanggal'])
    series_zat = series_zat.set_index('tanggal')

    X_zat = series_zat
    X_zat = series_zat.mean(axis=1)
    X_zat = np.reshape(X_zat.values, (len(X_zat), 1))
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_zat = scaler.fit_transform(X_zat)
    # train_size_zat = int(len(X_zat) * 0.8)
    train_size_zat = int(len(X_zat) * train_size[zat_selected])

    train_data_zat = X_zat[0:int(train_size_zat), :]
    x_train = []
    y_train = []

    for i in range(30, len(train_data_zat)):
        x_train.append(train_data_zat[i-30:i, 0])
        y_train.append(train_data_zat[i, 0])

        # Convert the x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    param = set_param_SVR[zat_selected][stasiun_selected]
    model = (param['Save_model'])
    model.fit(x_train, y_train)
    #regr_zat.fit(X_train_zat, Y_train_zat)

    test_data_30_zat = X_zat[train_size_zat - 30:, :]
    # Create the data sets x_test and y_test
    x_test_30_zat = []
    y_test_30_Pm10 = X_zat[train_size_zat:, :]
    for i in range(30, len(test_data_30_zat)):
        x_test_30_zat.append(test_data_30_zat[i-30:i, 0])

    # Convert the data to a numpy array
    x_test_30_zat = np.array(x_test_30_zat)
    # Reshape the data
    x_test_30_zat = np.reshape(
        x_test_30_zat, (x_test_30_zat.shape[0], x_test_30_zat.shape[1], 1))

    # Get the models predicted price values
    predictions_30_zat = model.predict(x_test_30_zat)

    hasil_zat = model.predict(x_test_30_zat[:+len(x_test_30_zat)])
    hasil_zat = hasil_zat.reshape(-1, 1)
    hasil_zat = scaler.inverse_transform(hasil_zat)[:, 0]
    print(hasil_zat)
    date_format = "%Y-%m-%d"
    first_date = series_zat.index[len(series_zat.index)-1]
    first_date += timedelta(days=1)
    day = 150
    predict_dates = pd.date_range(
        start=first_date, periods=day, freq='1d').tolist()

    n = []
    for d in predict_dates:
        n.append(d.date())

    df_forecast_zat = pd.DataFrame(
        {'tanggal': np.array(n), zat_selected: hasil_zat[0:len(n)]})
    df_forecast_zat['tanggal'] = pd.to_datetime(df_forecast_zat['tanggal'])
    df_forecast_zat = df_forecast_zat.set_index('tanggal')
    df_forecast_zat

    plt.figure(figsize=(14, 8))
    plt.plot(df_forecast_zat, label='Forecasted', color='orange')
    plt.ylabel(zat_selected)
    plt.xlabel("Year")
    plt.legend()
    plt.title("Peramalan "+zat_selected + " "+stasiun_selected)
    plt.savefig('static/gambar/grafik_zat.png')
    return render_template('predict.html', n=n, stasiun_selected=stasiun_selected, hasil_zat=hasil_zat, zat_selected=zat_selected)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=True)
