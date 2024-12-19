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
# Apply Custom CSS for Colorful Design
st.markdown(
    """
    <style>
    body {
        background-color: #f7f9fc;
        font-family: 'Arial', sans-serif;
        font-size: 14px; 
    }
    .main {
        background: linear-gradient(145deg, #ffffff, #e6f7ff);
        border-radius: 15px;
        padding: 20px;
    }
    h1 {
        color: #0056b3;
        text-shadow: 1px 1px 2px #a0c4ff;
        font-size: 2.5em;
    }
    h2, h3, h4 {
        color: #008080;
        text-shadow: 1px 1px 2px #b0e0e6;
    
    }
    .stButton > button {
        background: linear-gradient(to right, #4CAF50, #008CBA);
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 10px;
        font-size: 16px;
        transition: 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(to right, #66BB6A, #0093E9);
        transform: scale(1.05);
    }
    .small-btn button {
        font-size: 14px !important;
        padding: 5px 10px !important;
        border-radius: 5px !important;
    }
    .stSelectbox {
        border: 1px solid #87ceeb;
        border-radius: 21px;
        padding: 10px;
        font-size: 18px;
    }
    .stMarkdown {
        font-size: 14px;
        color: #333;
    }
    .stInfo {
        background-color: #e7f9e7;
        border-left: 5px solid #4CAF50;
    }
    .stWarning {
        background-color: #fff3cd;
        border-left: 5px solid #ffcc00;
    }
    </style>
    """,
    unsafe_allow_html=True
)
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
features_path = r"E:\third year\Big_Data_Tech\Project\Disease_prediction\features.json"
sorted_label_path = r"E:\third year\Big_Data_Tech\Project\Disease_prediction\sorted_label.json"
try:
    with open(features_path, "r") as file:
        features = json.load(file)
except FileNotFoundError:
    features = []
    st.error("Error: features.json file not found!")
except json.JSONDecodeError as e:
    features = []
    st.error(f"Error decoding JSON: {e}")

try:
    with open(sorted_label_path, "r") as file:
        sorted_label = json.load(file)
except FileNotFoundError:
    sorted_label = []
    st.error("Error: sorted_label.json file not found!")
except json.JSONDecodeError as e:
    sorted_label = []
    st.error(f"Error decoding JSON: {e}")

# Load the saved models
rf_model_path = r"E:\third year\Big_Data_Tech\Project\Disease_prediction\models\Random Forest"
dc_model_path = r"E:\third year\Big_Data_Tech\Project\Disease_prediction\models\Decision Tree"
nb_model_path = r"E:\third year\Big_Data_Tech\Project\Disease_prediction\models\Naive Bayes"

try:
    rf_model = RandomForestClassificationModel.load(rf_model_path)
    dc_model = DecisionTreeClassificationModel.load(dc_model_path)
    #nb_model = NaiveBayesModel.load(nb_model_path)
    st.success("Models loaded successfully!")
except Exception as e:
    st.error(f"Error loading models: {e}")
    rf_model, dc_model, nb_model = None, None, None

# Create a dictionary for numeric-to-label conversion
numeric_to_label = {i: label for i, label in enumerate(sorted_label)}  # Replace `features` with the actual label list if different
# Define a function to map numeric predictions to their corresponding labels
def get_label_from_numeric(numeric_value):
    return numeric_to_label.get(int(numeric_value), "Unknown Disease")

# Main Content
st.subheader("Select Your Symptoms")
# Initialize the session state to track symptoms
if "symptoms_count" not in st.session_state:
    st.session_state.symptoms_count = 4
# Create dropdowns dynamically
selected_symptoms = []
for i in range(st.session_state.symptoms_count):
    symptom = st.selectbox(
        f"Symptom {i+1}", options=[""] + features, key=f"symptom_{i}"
    )
    if symptom:
        selected_symptoms.append(symptom)

# Add and delete buttons with custom styling
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("➕ Add Symptom", key="add_btn", help="Add another symptom"):
        st.session_state.symptoms_count += 1
with col2:
    if st.button("➖ Remove Symptom", key="remove_btn", help="Remove a symptom") and st.session_state.symptoms_count > 1:
        st.session_state.symptoms_count -= 1

# Predict Button
if st.button("Predict Your Disease"):
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
            #prediction_nb = nb_model.transform(assembled_df).select("prediction").collect()[0]["prediction"]

            prediction_rf_label = get_label_from_numeric(prediction_rf)
            #prediction_nb_label = get_label_from_numeric(prediction_nb)

        except Exception as e:
            st.error(f"Error during prediction: {e}")
            prediction_rf, prediction_dc = None, None, 

        # Display Results
        st.write("### 🩺 Prediction Results:")
        st.info(f"Your potential disease: {prediction_rf_label}")
        #st.info(f"**Naive Bayes Prediction:** {prediction_nb_label}")
        st.warning("⚠️ If symptoms worsen or do not improve, please consult a healthcare professional immediately.")

# Footer
st.markdown("---")
st.markdown("**Disclaimer:** This tool provides predictions based on limited input and is not a substitute for professional medical advice. Always consult a doctor for accurate diagnosis.")
