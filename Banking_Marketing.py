import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import numpy as np
import pickle


data = pd.read_csv('bank.csv')
# data


# # Transform the dataset to a Machine-Readeable Language

# def transformer(dataframe):
#     from sklearn.preprocessing import StandardScaler, LabelEncoder
#     scaler = StandardScaler()
#     encoder = LabelEncoder()

#     for i in dataframe.columns:
#         if dataframe[i].dtypes != 'O':
#             dataframe[i] = scaler.fit_transform(dataframe[[i]])
#         else:
#             dataframe[i] = encoder.fit_transform(dataframe[i])
#     return dataframe

# y = data.deposit
# x = data.drop(['deposit'], axis = 1)


# sel_features = ['duration', 'balance', 'day', 'age', 'month', 'pdays', 'job', 'campaign', 'education']

# new_frame = data[sel_features] #................................................... Turn the selected features to a dataframe

# model = pickle.load(open('BankMkt.pkl','rb'))




# ............Streamlit Development..................

model = pickle.load(open('BankMkt.pkl', "rb"))

st.markdown("<h1 style = 'color: #0A2647; text-align: center; font-family: helvetica '>BANKING MARKETING CAMPAIGN</h1>", unsafe_allow_html = True)

st.markdown("<h4 style = 'margin: -30px; color: #FF87CA; text-align: center; font-family: cursive '>Built By Olukini Kunle</h4>", unsafe_allow_html = True)

st.image('pngwing.com (1).png', width = 500)

st.markdown("<h2 style = 'color: #ED7D31; text-align: center; font-family: montserrat '>BACKGROUND HISTORY</h2>", unsafe_allow_html = True)

st.markdown("<p>Bank marketing is the practice of attracting and acquiring new customers through traditional media and digital media strategies. The use of these media strategies helps determine what kind of customer is attracted to a certain institutions</p>", unsafe_allow_html = True)



password = ['one', 'two', 'three']
username = st.text_input('Pls enter your username')
passes = st.text_input('Pls input password')

if passes in password:
    st.toast('Registered User')
    print(f'Welcome {username}, Pls enjoy your usage as a registered user')
else:
    st.error('You are not a registered user. But you have three trials')

    st.sidebar.image('pngwing.com (2).png', caption = f'Welcome {username}')

    dx = data[['duration', 'balance', 'day', 'age', 'month', 'pdays', 'job', 'campaign', 'education']]
    st.write(dx.head())

    st.markdown('<br><br>', unsafe_allow_html= True)


# INPUT FEATURES
input_type = st.sidebar.radio("Select Your Preferred Input Style", ["Slider", "Number Input"])
st.markdown('<br><br>', unsafe_allow_html= True)

if input_type == 'Slider':
    duration = st.sidebar.slider('duration', data['duration'].min(), data['duration'].max())
    balance = st.sidebar.slider('balance', data['balance'].min(), data['balance'].max())
    day = st.sidebar.slider('day', data['day'].min(), data['day'].max())
    age = st.sidebar.slider('age', data['age'].min(), data['age'].max())
    month = st.sidebar.selectbox('month', data['month'].unique())
    pdays = st.sidebar.slider('pdays', data['pdays'].min(), data['pdays'].max())
    job = st.sidebar.selectbox('job', data['job'].unique())
    campaign = st.sidebar.slider('campaign', data['campaign'].min(), data['campaign'].max())
    education = st.sidebar.selectbox('education', data['education'].unique())
else:
    duration = st.sidebar.number_input('duration', data['duration'].min(), data['duration'].max())
    balance = st.sidebar.number_input('balance', data['balance'].min(), data['balance'].max())
    day = st.sidebar.number_input('day', data['day'].min(), data['day'].max())
    age = st.sidebar.number_input('age', data['age'].min(), data['age'].max())
    month = st.sidebar.selectbox('month', data['month'].unique())
    pdays = st.sidebar.number_input('pdays', data['pdays'].min(), data['pdays'].max())
    job = st.sidebar.selectbox('job', data['job'].unique())
    campaign = st.sidebar.number_input('campaign', data['campaign'].min(), data['campaign'].max())
    education = st.sidebar.selectbox('education', data['education'].unique())

st.header('Input Values')
# Bring all the inputs into a dataframe
input_variable = pd.DataFrame([{'duration':duration, 'balance': balance, 'day': day, 'age':age, 'month':month, 'pdays': pdays, 'job':job, 'campaign': campaign, 'education': education}])

st.write(input_variable)

# Standard Scale the Input Variable.
for i in input_variable.columns:
    scaler = StandardScaler()
    encoder = LabelEncoder()

    if input_variable[i].dtypes != 'O':
            input_variable[i] = scaler.fit_transform(input_variable[[i]])
    else:
            input_variable[i] = encoder.fit_transform(input_variable[i])
    

st.markdown('<hr>', unsafe_allow_html=True)
st.markdown("<h2 style = 'color: #0A2647; text-align: center; font-family: helvetica '>Model Report</h2>", unsafe_allow_html = True)

if st.button('Press To Predict'):
    predicted = model.predict(input_variable)
    st.toast('Profitability Predicted')
    st.image('pngwing.com (6).png', width = 200)
    st.write(f'The predicted value is:{predicted}')
    # st.success('Predicted. Pls check the Interpretation Tab for interpretation')

    st.markdown("<h2 style = 'color: #132043; text-align: center; font-family: montserrat '>Model Interpretation</h2>", unsafe_allow_html = True)

    if predicted== 1:
        st.success('The client subscribed to the product')
    else:
        st.success('the client does not subscribed to the product')


