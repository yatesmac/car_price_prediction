'''
predict.py - Flask application:
Loads the XGBoost and Artificial Neural Network models and evaluates given data.
'''
import pickle
import logging
import sys
import json

from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify

# sys.path lists directories that Python searches for modules to import
sys.path.append('../data') 
from data_preparation import dataset


logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler('../../logs/debug.log', mode='a'),
        logging.StreamHandler(sys.stdout)]
    ) 


def load_data(test_data):
    root = '../../data' # Raw Data root director
    train_csv = f'{root}/train/df_full_train.csv'
    target = 'price'
    _, _, X_test, y_test = dataset(train_csv, test_data, target, scaler=True)
    return X_test, y_test


def load_pickle(model_file):
    '''Load the pre-trained pickle model and other necessary components'''
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    logger.info('Successfully loaded Pickle Model.')
    return model


def load_keras(model_file, weights=None):
    model = load_model(model_file)
#   model.load_weights(weights)
    logger.info('Successfully loaded Keras Model.')
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
    model_file_xgb = f'{root}/xgb_v1.pkl'
    model_xgb = load_pickle(model_file_xgb)


    model_file_rf = f'{root}/rf_v1.pkl'
    model_rf = load_pickle(model_file_rf)

    model_file_ann = f'{root}/ann_v1.h5'
    # weights = f'{root}/checkpoints/ann_v1_.ckpt'
    model_ann = load_keras(model_file_ann)

    models = [
        ('RF', model_rf),
        ('XGB', model_xgb),
        ('ANN', model_ann)
    ]

    data = request.get_json()
    data = json.loads(data)
    X, y = load_data(test_data=data['url'])
    results = {'ACTUAL': float(y)}
    for name, model in models:
        results[name] = evaluate(model, X)
        
    return jsonify(results)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9696)