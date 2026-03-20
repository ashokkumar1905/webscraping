from flask import Flask, request, render_template
from model import SalesPredictor
import pandas as pd
import numpy as np

app = Flask(__name__)

# Initialize your SalesPredictor and load the models
predictor = SalesPredictor("train.csv")
predictor.load_models()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    date = request.form['date']
    store = request.form['store']
    item = request.form['item']

    # Prepare the input data for prediction
    date_obj = pd.to_datetime(date)

    # Extract features from the input date
    input_data = pd.DataFrame({
        'store': [store],
        'item': [item],
        'year': [date_obj.year],
        'month': [date_obj.month],
        'day': [date_obj.day],
        'm1': [np.sin(date_obj.month * (2 * np.pi / 12))],
        'm2': [np.cos(date_obj.month * (2 * np.pi / 12))]
    })

    # Make predictions
    linear_prediction, xgb_prediction = predictor.predict(input_data)

    return render_template('result.html', 
                           date=date, 
                           store=store, 
                           item=item, 
                           linear_prediction=linear_prediction[0], 
                           xgb_prediction=xgb_prediction[0])

if __name__ == "__main__":
    app.run(debug=True)
