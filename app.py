from flask import Flask, render_template, request
import joblib
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__, template_folder='templates')

# Load the saved models

Naive_Bayes_Algorithm = joblib.load("nb_model.pkl")
LGBM_Classifier_Algorithm = joblib.load("clf_model.pkl")

@app.route('/', methods=['GET', 'POST'])
def index():
    predictions = {}
    if request.method == 'POST':
        input_data = request.form['input_data']
        input_data_list = [float(x.strip()) for x in input_data.split(',')]

        # Perform predictions using different models

        predictions['Random_Forest'] = 'Safe' if make_prediction(input_data_list, Naive_Bayes_Algorithm) == 0 else 'Malicious'
        predictions['k_nearest_neighbor'] = 'Safe' if make_prediction(input_data_list, LGBM_Classifier_Algorithm) == 0 else 'Malicious'
        predictions['naive_bayes'] = 'Safe' if make_prediction(input_data_list, Naive_Bayes_Algorithm) == 0 else 'Malicious'
        predictions['ANN'] = 'Safe' if make_prediction(input_data_list, LGBM_Classifier_Algorithm) == 0 else 'Malicious'

        predictions['LGBM_clasifier'] = 'Safe' if make_prediction(input_data_list, LGBM_Classifier_Algorithm) == 0 else 'Malicious'

        print("Predictions:", predictions)

        return render_template('index.html', predictions=predictions)

    return render_template('index.html')

def make_prediction(input_data, model):
    try:
        # Perform the prediction using the model
        input_data_reshaped = np.array(input_data).reshape(1, -1)
        prediction = model.predict(input_data_reshaped)
        print("Prediction:", prediction)

        return prediction[0]
    except ValueError:
        # Handle the case where input data cannot be converted to numbers
        print(ValueError)
        return None

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
