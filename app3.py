from pyspark import SparkConf, SparkContext
from flask import Flask, render_template, request
from pyspark.sql import SparkSession
from pyspark.ml.classification import DecisionTreeClassificationModel, RandomForestClassificationModel
from pyspark.ml.linalg import Vectors
from pyspark.ml import PipelineModel

# Initialize Spark session
# spark = SparkSession.builder \
#     .appName("ModelServing") \
#     .getOrCreate()

conf = SparkConf()
conf.set("spark.python.worker.timeout", "600")  # Increase the timeout to 600 seconds
conf.set("spark.network.timeout", "600s")
sc = SparkContext(conf=conf)

# Load PySpark models
loaded_pipeline_model = PipelineModel.load(r"E:\third year\Big_Data_Tech\Project\Disease_prediction\models")
#random_forest_model = PipelineModel.load(r"E:\third year\Big_Data_Tech\Project\Disease_prediction\models\Random Forest")

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')  # Render the input form

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input features from the form and convert them to floats
        features = []
        for i in range(1, 101):  # Adjust the range based on the number of input features your model expects
            feature_value = request.form[f'feature{i}']
            features.append(float(feature_value))  # Convert input to float
            
        # Convert features into a PySpark DataFrame
        df = spark.createDataFrame([(0, Vectors.dense(features))], ["id", "features"])
        
        # Make predictions using the models
        dt_prediction = loaded_pipeline_model.transform(df).collect()[0].prediction
        rf_prediction = loaded_pipeline_model.transform(df).collect()[0].prediction

        # Render results back to the user
        return render_template(
            'index.html',
            dt_prediction=dt_prediction,
            rf_prediction=rf_prediction,
            features=features
        )
    
    except ValueError as e:
        # Handle any conversion errors gracefully
        return render_template('index.html', error="Invalid input. Please enter numeric values.")
    
    except Exception as e:
        # Catch other exceptions
        return render_template('index.html', error=str(e))


if __name__ == '__main__':
    app.run(debug=True)
