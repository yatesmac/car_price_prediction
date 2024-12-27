'''predict.py - Flask application'''
import pickle

from flask import Flask, request, jsonify


def load(model_file):
    '''Load the pre-trained model and other necessary components'''
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    return model


def train(X_data, y_data):
    '''Process the data, make predictions using the model, and return the results'''
    prediction = model.predict(X_data)

    return {
        'actual_value': y_data,
        'prediction': float(prediction)
    }


app = Flask(__name__)
model_file = '../../models/xgb.pkl'
model = load(model_file=model_file)


# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    results = train(data=data)
    return jsonify(results)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9696)