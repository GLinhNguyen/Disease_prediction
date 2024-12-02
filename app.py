import streamlit as st
import pandas as pd
import numpy as np
import nbformat
from notebook import *
from nbconvert import PythonExporter
# Load the notebook file
with open(r'e:/third year/Big_Data_Tech/Project/disease_project/notebook.ipynb') as f:
    notebook_content = f.read()

# Convert the notebook to Python code
notebook_node = nbformat.reads(notebook_content, as_version=4)
exporter = PythonExporter()
source, _ = exporter.from_notebook_node(notebook_node)

# Execute the notebook code
exec(source)
# Example of a text input
user_input1 = st.text_input('Enter :')

# Slider for numeric input
slider_value = st.slider('Pick a number', 0, 100, 50)


