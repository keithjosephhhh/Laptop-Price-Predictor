import streamlit as st
import numpy as np
import pickle
import math


pipe = pickle.load(open("pipe.pkl", 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

st.title("Laptop Price Predictor")

Company = st.selectbox('Brand', df['Company'].unique())
Type = st.selectbox("Type", df['TypeName'].unique())
Ram = st.selectbox("Ram (in GB)", [2,4,6,8,12,16,24,32,64])

Weight = st.number_input('Weight of the Laptop')

TouchScreen = st.selectbox("Touch Screen", ['Yes', 'No'])
IPS = st.selectbox("IPS", ['Yes', 'No'])

screen_size = st.number_input('Screen Size')

Resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])
cpu = st.selectbox('CPU',df['Cpu brand'].unique())
hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])
ssd = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])
gpu = st.selectbox('GPU',df['Gpu brand'].unique())
os = st.selectbox('OS',df['os'].unique())

if st.button('Predict Price'):
    ppi = None
    if screen_size < 1:
        screen_size = 1

    if TouchScreen == 'Yes':
        TouchScreen = 1
    else:
        TouchScreen = 0

    if IPS == "Yes":
        IPS = 1
    else:
        IPS = 0

    X_res = int(Resolution.split('x')[0])
    Y_res = int(Resolution.split("x")[1])


    ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size
    query = np.array([Company, Type, Ram, Weight, TouchScreen, IPS, ppi, cpu, hdd, ssd, gpu, os])

    query = query.reshape(1,12)
    st.title("The predicted price of this configuration is " + str(int(np.exp(pipe.predict(query)[0]))))
