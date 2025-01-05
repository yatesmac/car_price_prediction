'''
predict.py - Flask application:
Loads the XGBoost and Artificial Neural Network models and evaluates given data.
'''
import pickle

from keras.models import load_model
from flask import Flask, request, jsonify


def load_xgb(model_file):
    '''Load the pre-trained model and other necessary components'''
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    return model


def load_ann(model_file, weights=None):
    model = load_model(model_file)
#   model.load_weights(weights)
    return model


def evaluate(model, data):
    '''Process the data, make predictions using the model, and return the results'''
    prediction = model.predict(data)
    return float(prediction)


app = Flask(__name__)

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():

    root = '../../models' # Models root directory
    model_file_xgb = f'{root}/xgb.pkl'
    model_xgb = load_xgb(model_file_xgb)

    model_file_ann = f'{root}/ann_v1.h5'
    weights = f'{root}/checkpoints/ann_v1_.ckpt' # TODO: Specify weights
    model_ann = load_ann(model_file_ann, weights)

    models = [
        ('XGB', model_xgb),
        ('ANN', model_ann)
    ]

    data = request.get_json()
    results = {}
    for name, model in models:
        results[name] = evaluate(model, data)
        
    return jsonify(results)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9696)