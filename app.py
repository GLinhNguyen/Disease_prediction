import pandas as pd
import numpy as np
import json
import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import CrossValidatorModel
from pyspark.sql.types import StructType, StructField, StringType

st.title("Disease Prediction")
st.write("Select symptoms to predict the disease.")

# Initialize Spark session
spark = SparkSession.builder \
    .appName("DiseasePrediction") \
    .config("spark.executor.memory", "2g") \
    .config("spark.driver.memory", "2g") \
    .config("spark.driver.host", "127.0.0.1") \
    .getOrCreate()

# Path to top features JSON file
top_features_path = r"E:\third year\Big_Data_Tech\Project\Disease_prediction\top_features.json"
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
rf_model_path = r"E:\third year\Big_Data_Tech\Project\Disease_prediction\models\Random Forest"
try:
    rf_model = CrossValidatorModel.load(rf_model_path)
    st.success("Model loaded successfully!")
except Exception as e:
    rf_model = None
    st.error(f"Error loading model: {e}")

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
        # Create a dictionary for the selected symptoms
        symptoms_dict = {symptom: 1 for symptom in selected_symptoms}

        # Convert selected symptoms into a Spark DataFrame
        schema = StructType([StructField(symptom, StringType(), True) for symptom in selected_symptoms])
        symptoms_df = spark.createDataFrame([symptoms_dict], schema=schema)

        # Assemble features
        assembler = VectorAssembler(inputCols=selected_symptoms, outputCol="features")
        symptoms_features_df = assembler.transform(symptoms_df)

        # Make predictions
        predictions = rf_model.transform(symptoms_features_df)

        # Extract the predicted label
        predicted_label = predictions.select("prognosis").collect()[0]["prognosis"]

        # Display the result
        st.write("Your predicted disease is:", predicted_label)

# Stop the Spark session
spark.stop()

