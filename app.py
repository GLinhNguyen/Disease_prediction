# import streamlit as st
# import pandas as pd
# import numpy as np
# import nbformat
# from pyspark.sql import SparkSession
# from pyspark.ml.feature import VectorAssembler
# from notebook import *
# from nbconvert import PythonExporter
# # Load the notebook file
# with open(r'e:/third year/Big_Data_Tech/Project/Disease_prediction/notebook.ipynb') as f:
#     notebook_content = f.read()

# # Convert the notebook to Python code
# notebook_node = nbformat.reads(notebook_content, as_version=4)
# exporter = PythonExporter()
# source, _ = exporter.from_notebook_node(notebook_node)

# # Execute the notebook code
# execution_context = {}
# exec(source, execution_context)
# top_features = execution_context.get("top_features")

# rf_model = execution_context.get("rf_model", None)
# # Initialize Spark session
# spark = SparkSession.builder.appName("DiseasePrediction").getOrCreate()

# # Streamlit UI

# # Streamlit UI
# st.title("Disease Prediction")

# # Feature selection dropdowns
# selected_symptoms = []
# for i in range(1, 6):  # For symptoms 1 to 7
#     selected_symptom = st.selectbox(f"Symptom {i}",   options=[""] + top_features, key=f"symptom_{i}")
#     selected_symptoms.append(selected_symptom)

# # Predict button
# # Predict button
# if st.button("Predict your disease"):
#     # Check if all symptoms are selected (not blank)
#     if "" in selected_symptoms:
#         st.warning("Please select all symptoms before predicting.")
#     else:
#         # Convert selected symptoms into a Spark DataFrame
#         symptoms_df = spark.createDataFrame(
#             [[selected_symptoms]], schema=top_features
#         )

#         # Assemble features
#         assembler = VectorAssembler(inputCols=top_features, outputCol="features")
#         symptoms_features_df = assembler.transform(symptoms_df)

#         # Make predictions
#         predictions = rf_model.transform(symptoms_features_df)

#         # Extract the predicted label
#         predicted_label = predictions.select("prediction").collect()[0]["prediction"]

#         # Display the result
#         st.write("Your disease is:", predicted_label)

  


import streamlit as st
import pandas as pd
import numpy as np
import nbformat
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from nbconvert import PythonExporter

# Load the notebook file
notebook_file_path = r'e:/third year/Big_Data_Tech/Project/Disease_prediction/notebook.ipynb'

# Function to convert notebook to Python code
def convert_notebook_to_python(notebook_file_path):
    with open(notebook_file_path, "r") as f:
        notebook_content = f.read()
    notebook_node = nbformat.reads(notebook_content, as_version=4)
    exporter = PythonExporter()
    source, _ = exporter.from_notebook_node(notebook_node)
    return source

# Execute the notebook code to extract variables
def execute_notebook_code(source):
    execution_context = {}
    exec(source, execution_context)
    return execution_context

# Streamlit UI
st.title("Disease Prediction")

# Initialize Spark session
spark = SparkSession.builder.appName("DiseasePrediction").getOrCreate()

# Convert notebook to Python and execute the code to get variables
source = convert_notebook_to_python(notebook_file_path)
execution_context = execute_notebook_code(source)

# Extract top features and the trained model from the executed notebook
top_features = execution_context.get("top_features", [])
rf_model = execution_context.get("rf_model", None)

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

