import pandas as pd
import numpy as np
import json
import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassificationModel, DecisionTreeClassificationModel

from pyspark import SparkContext, SparkConf
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

st.title("Disease Prediction")
st.write("Select symptoms to predict the disease.")

# Check if SparkContext already exists
if not SparkContext._active_spark_context:
    conf = SparkConf().setAppName("Disease_Prediction").setMaster("local[*]")
    sc = SparkContext(conf=conf)

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Disease_Prediction") \
    .getOrCreate()


# Path to top features JSON file
top_features_path = r"E:\third year\Big_Data_Tech\Project\Disease_prediction\features.json"
try:
    with open(top_features_path, "r") as file:
        features = json.load(file)
except FileNotFoundError:
    features = []
    st.error("Error: top_features.json file not found!")
except json.JSONDecodeError as e:
    features = []
    st.error(f"Error decoding JSON: {e}")
import shutil
import os

# Specify Spark temp directories
temp_dirs = ['/tmp/spark-local', '/tmp/spark-temp']

for d in temp_dirs:
    if os.path.exists(d):
        shutil.rmtree(d)

# Load the saved model
rf_model_path = r"E:\third year\Big_Data_Tech\Project\Disease_prediction\models\Random Forest"
nb_model_path = r"E:\third year\Big_Data_Tech\Project\Disease_prediction\models\Naive Bayes"
try:
    rf_model = RandomForestClassificationModel.load(rf_model_path)
    
    st.success("Model loaded successfully!")
except Exception as e:
    rf_model = None
    st.error(f"Error loading model: {e}")

# Allow user to select 5 symptoms from the list
selected_symptoms = []
for i in range(5):  # Allow selection of 5 symptoms only
    symptom = st.selectbox(f"Symptom {i+1}", options=[""] + features, key=f"symptom_{i}")
    if symptom:
        selected_symptoms.append(symptom)

# Make prediction based on selected symptoms
if st.button("Predict your disease"):
    if len(selected_symptoms) == 0:
        st.warning("Please select at least one symptom.")
    else:
        # Map selected symptoms to binary values (1 for selected, 0 for not selected)
        symptoms_dict = {symptom: 1 if symptom in selected_symptoms else 0 for symptom in features}

        # # Convert the dictionary to a DataFrame
        # symptoms_df = pd.DataFrame([symptoms_dict])

        # Convert selected symptoms into a Spark DataFrame
        schema = StructType([StructField(symptom, IntegerType(), True) for symptom in selected_symptoms])
        symptoms_df = spark.createDataFrame([symptoms_dict], schema=schema)

        # Assemble features
        assembler = VectorAssembler(inputCols=selected_symptoms, outputCol="features")
        symptoms_features_df = assembler.transform(symptoms_df)

        # Make predictions
        predictions = rf_model.transform(symptoms_features_df)

        # Extract the predicted label
        predicted_label = predictions.select("prediction").collect()[0]["prognosis"]
        # Display the result
        st.write("Your predicted disease is:", predicted_label)

# Stop the Spark session
spark.stop()
