# 1. Import Packages
import os, mlflow
import mlflow.sklearn
import pickle as pkl
import pandas as pd
import numpy as np
from sklearn import metrics, model_selection, preprocessing, pipeline
import streamlit as st

# 2. Load in your resources
# Use cache
# (A) Function to load pickle object
@st.cache_resource  # Cache resourse :--> when streamlit run it will look to these resources once and save the time
def load_pickle(filepath):
    with open(filepath,'rb') as f:
        pickle_object = pkl.load(f)
    return pickle_object

# (B) Function to load machine learning model
@st.cache_resource
def load_model(uri):
    model = mlflow.sklearn.load_model(uri)
    return model

# Use the functions to load in the resources
os.chdir(r"C:\Users\001057\Desktop\ML_Ciast\batch_1\assessment")
# (A) Load ordinal encoder
encoder = load_pickle("src/ordinal_encoder.pkl")

# (B) Load the model
model = load_model("models:/diamond_assessment_model@champion")

# Add a Title

st.title("Diamond Price Predictor")

# Create the input widgets for user inputs
# Form
with st.form("User Inputs"):

    # (A) carat
    carat = st.number_input("Carat",min_value=0.0,max_value=6.0, step=0.01)

    # (B) cut
    cut_group = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
    cut = st.selectbox("Cut",options=cut_group)

    # (C) color
    color_group = ["D","E","F","G","H","I","J"]
    color = st.selectbox("Color", options=color_group)

    # (D) Clarity
    clarity_group = ['I1', 'IF', 'SI1', 'SI2', 'VS1', 'VS2', 'VVS1', 'VVS2']
    clarity = st.selectbox("Clarity", options=clarity_group)

    # (E) depth
    depth = st.number_input("Depth", min_value=40.0, max_value=80.0, step=0.01)

    # (F) table
    table = st.number_input("Table", min_value=40, max_value=100)

    # Form submit button
    submit = st.form_submit_button("Submit")

columns = ['Carat','Cut','Color','Clarity','Depth',"Table"]
user_input = pd.DataFrame(np.array([[carat,cut,color,clarity,depth,table]]), columns=columns)

# Process the categorical inputs
user_input[['Cut', 'Color','Clarity']] = encoder.transform(user_input[['Cut','Color','Clarity']].values)
st.write(user_input)

prediction = model.predict(user_input.values)
# prediction_class = label_map[prediction[0]]
st.write("Your Diamond Price is")
st.write(prediction[0])
