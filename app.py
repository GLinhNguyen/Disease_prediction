import pandas as pd
import numpy as np
import nbformat
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import CrossValidatorModel
from nbconvert import PythonExporter
import json
import streamlit as st
st.title("Disease Prediction")
st.write("Select symptoms to predict the disease.")

# Initialize Spark session
spark = SparkSession.builder.appName("DiseasePrediction").getOrCreate()

top_features_path = "/Users/baonguyen/Desktop/Disease_prediction/top_features.json"
try:
    with open(top_features_path, "r") as file:
        top_features = json.load(file)
except FileNotFoundError:
    top_features = []
    st.error("Error: top_features.json file not found!")
except json.JSONDecodeError as e:
    top_features = []
    st.error(f"Error decoding JSON: {e}")

# Load the saved model
rf_model_path = "/Users/baonguyen/Desktop/Disease_prediction/rf_model"
try:
    rf_model = CrossValidatorModel.load(rf_model_path)
    st.success("Model loaded successfully!")
except Exception as e:
    rf_model = None
    st.error(f"Error loading model: {e}")

if not top_features or not rf_model:
    st.error("Model or top features not found in the notebook.")
else:
    # Feature selection dropdowns
    selected_symptoms = []
    for i in range(1, 6):  # For symptoms 1 to 5 (or however many features)
        selected_symptom = st.selectbox(f"Symptom {i}", options=[""] + top_features, key=f"symptom_{i}")
        selected_symptoms.append(selected_symptom)

    # Predict button
    if st.button("Predict your disease"):
        # Check if all symptoms are selected (not blank)
        if "" in selected_symptoms:
            st.warning("Please select all symptoms before predicting.")
        else:
            # Convert selected symptoms into a Spark DataFrame
            symptoms_df = spark.createDataFrame([[selected_symptoms]], schema=selected_symptoms)

            # Assemble features 
            assembler = VectorAssembler(inputCols=top_features, outputCol="features")
            symptoms_features_df = assembler.transform(symptoms_df)

            # Make predictions
            predictions = rf_model.transform(symptoms_features_df)

            # Extract the predicted label
            predicted_label = predictions.select("prognosis").collect()[0]["prognosis"]

            # Display the result
            st.write("Your predicted disease is:", predicted_label)