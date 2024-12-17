import pandas as pd
import numpy as np
import json
import streamlit as st
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import (
    DecisionTreeClassificationModel, 
    RandomForestClassificationModel,
    NaiveBayesModel
)
from pyspark.sql.types import StructType, StructField, IntegerType

# Streamlit Page Configuration
st.set_page_config(page_title="Disease Prognosis", page_icon="🩺", layout="wide")

# Web Design
st.title("🩺 Disease Prognosis System")
st.subheader("This web is designed for light disease prognosis. **Please consult a doctor** if symptoms are heavy or persistent.")

# Stop SparkContext if it exists
if SparkContext._active_spark_context:
    SparkContext._active_spark_context.stop()

# Create SparkSession
spark = SparkSession.builder \
    .appName("DiseasePrediction") \
    .master("local[*]") \
    .getOrCreate()

# Load Features
features_path = "/content/drive/MyDrive/Colab Notebooks/Big data/features.json"
try:
    with open(features_path, "r") as file:
        features = json.load(file)
except FileNotFoundError:
    features = []
    st.error("Error: features.json file not found!")
except json.JSONDecodeError as e:
    features = []
    st.error(f"Error decoding JSON: {e}")

# Load the saved models
rf_model_path = "/content/drive/MyDrive/Colab Notebooks/Big data/Random Forest"
dc_model_path = "/content/drive/MyDrive/Colab Notebooks/Big data/Decision Tree"
nb_model_path = "/content/drive/MyDrive/Colab Notebooks/Big data/Naive Bayes"

try:
    rf_model = RandomForestClassificationModel.load(rf_model_path)
    dc_model = DecisionTreeClassificationModel.load(dc_model_path)
    nb_model = NaiveBayesModel.load(nb_model_path)
    st.success("All models loaded successfully!")
except Exception as e:
    st.error(f"Error loading models: {e}")
    rf_model, dc_model, nb_model = None, None, None

# Allow user to select 5 symptoms
st.sidebar.header("Select Your Symptoms")
selected_symptoms = []
for i in range(5):  # Allow selection of up to 5 symptoms
    symptom = st.sidebar.selectbox(f"Symptom {i+1}", options=[""] + features, key=f"symptom_{i}")
    if symptom:
        selected_symptoms.append(symptom)

# Predict Button
if st.sidebar.button("Predict Your Disease"):
    if len(selected_symptoms) == 0:
        st.warning("Please select at least one symptom.")
    else:
        # Map selected symptoms to binary values (1 for selected, 0 for not selected)
        symptoms_dict = {symptom: 1 if symptom in selected_symptoms else 0 for symptom in features}

        # Convert dictionary to pandas DataFrame
        symptoms_df = pd.DataFrame([symptoms_dict])

        # Convert pandas DataFrame to Spark DataFrame
        schema = StructType([StructField(symptom, IntegerType(), True) for symptom in features])
        spark_df = spark.createDataFrame(symptoms_df, schema)

        # Assemble features
        assembler = VectorAssembler(inputCols=features, outputCol="features")
        assembled_df = assembler.transform(spark_df)

        # Make Predictions
        try:
            prediction_rf = rf_model.transform(assembled_df).select("prediction").collect()[0]["prediction"]
            prediction_dc = dc_model.transform(assembled_df).select("prediction").collect()[0]["prediction"]
            prediction_nb = nb_model.transform(assembled_df).select("prediction").collect()[0]["prediction"]
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            prediction_rf, prediction_dc, prediction_nb = None, None, None

        # Display Results
        st.write("### 🩺 Prediction Results:")
        st.info(f"**Random Forest Prediction:** {prediction_rf}")
        st.info(f"**Decision Tree Prediction:** {prediction_dc}")
        st.info(f"**Naive Bayes Prediction:** {prediction_nb}")
        st.warning("⚠️ If symptoms worsen or do not improve, please consult a healthcare professional immediately.")

# Footer
st.markdown("---")
st.markdown("**Disclaimer:** This tool provides predictions based on limited input and is not a substitute for professional medical advice. Always consult a doctor for accurate diagnosis.")
