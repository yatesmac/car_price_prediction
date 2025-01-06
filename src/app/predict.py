'''
predict.py - Flask application:
Loads the XGBoost and Artificial Neural Network models and evaluates given data.
'''
import pickle
import logging
import sys

from keras.models import load_model
from flask import Flask, request, jsonify


logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler('../../logs/debug.log', mode='a'),
        logging.StreamHandler(sys.stdout)]
    ) 


def load_xgb(model_file):
    '''Load the pre-trained model and other necessary components'''
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    logger.info('Successfully loaded XGB Model.')
    return model


def load_ann(model_file, weights=None):
    model = load_model(model_file)
#   model.load_weights(weights)
    logger.info('Successfully loaded ANN Model.')
    return model


def evaluate(model, data):
    '''Process the data, make predictions using the model, and return the results'''
    prediction = model.predict(data)
    logger.info('Evaluating...')
    return float(prediction)


app = Flask(__name__)

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():

    root = '../../models' # Models root directory
    model_file_xgb = f'{root}/xgb.pkl'
    model_xgb = load_xgb(model_file_xgb)

    model_file_ann = f'{root}/ann_v1.keras'
    # weights = f'{root}/checkpoints/ann_v1_.ckpt'
    model_ann = load_ann(model_file_ann)

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