
# Car Price Prediction

![Dataset Cover](/img/dataset_cover.webp)

<!-- TOC -->
- [Project Description](#project-description)
- [About the Dataset](#about-the-dataset)

    - [Column Descriptions](#column-descriptions)
    - [Key Features](#key-features)
    - [Target Variable](#target-variable)

- [Project Structure](#project-structure)
- [Project Setup](#project-setup)

    - [To reproduce the project without Docker](#to-reproduce-the-project-without-docker)

- [Containerization](#containerization)

    - [To reproduce the project with Docker](#to-reproduce-the-project-with-docker)

- [Results and Evaluation](#results-and-evaluation)
- [Acknowledgements](#acknowledgements)
<!-- /TOC -->

## Project Description

The goal of this project is to predict the price of a car. This is a regression problem.

The project involves the following steps:

- exploring the data
- splitting the data into datasets to be used in training;
- initially training the following machine learning models:

    - Linear Regression,
    - Lasso Regression,
    - Ridge  Regression,
    - Decision Tree Regression,
    - Random Forest Regression,
    - XGBoost Regression,
    - Gradient Boosting Regression

- the top three models are selected for hyperparameter turning, namely:
     
    - Random Forest Regression,
    - XGBoost Regression, and
    - Gradient Boosting Regression

- evaluating the models and selecting the most performant one;
- in addition to these models, a neural network (Artificial Neural Net) is also trained for comparison
- creating a web application with the final model using Flask and Gunicorn;
- deploying the model locally with Docker.

## Project Structure

```bash
.
├── Dockerfile
├── README.md
├── data                                # CSV Datasets 
│   ├── external
│   │   └── final_scout_not_dummy.csv   : CSV dataset downloaded from Kaggle
│   ├── test                            
│   │   ├── df_test.csv                 : Testing Dataset
│   │   └── random_rows
│   │       ├── row_3.csv               : Random row sampled from Test CSV used with Flask App
│   │       └── row_59.json
│   └── train
│       ├── df_full_train.csv           : Training + Validation Dataset
│       ├── df_train.csv                : Training Dataset
│       └── df_val.csv                  : Validation Dataset
├── img
│   └── dataset_cover.webp
├── logs
│   └── debug.log
├── models                              # Models
│   ├── ann_v1.h5                       : Neural Network Model - saved with weights
│   ├── ann_v1.keras
│   ├── rf_v1.pkl                       : Random Forest Model
│   └── xgb_v1.pkl                      : XGBoost Model
├── notebooks                           # Notebooks
│   ├── 1_data_exploration.ipynb        : Data Exploration with non-persistent changes
│   ├── 2_data_preparation.ipynb        : Split Kaggle dataset to Training, Validation and Testing Datasets
│   ├── 3_skl_xgb_models.ipynb          : Training and tuning SKLearn and XGB models
│   └── 4_ann_model.ipynb               : Training and tuning neural network
├── pyproject.toml
├── requirements.txt
└── src                                 # Scripts
    ├── app
    │   ├── predict.py                  : Flask Application
    │   ├── sample.py                   : Sample a single row from Test CSV
    │   ├── test.py                     : Test Flask app using sampled data
    │   └── testing.ipynb
    ├── data
    │   ├── data_preparation.py         : Prepare Training and Testing datatsets 
    │   ├── flattencolumns.py           : Flatten multi-column columns for DictVectorization
    │   └── split_csv.py                : Split Kaggle dataset to Training, Validation and Testing Datasets 
    └── models
        ├── ann.py                      : Train and save Neural Net Model
        ├── rf.py                       : Train and save Random Forest Model
        └── xgb.py                      : Train and save XGB model
        
```

## Project Setup

Note: This project was developed using Ubuntu in Github Codespaces. As such the code snippets below work in Linux-based environments. The Microsoft Windows implementation may differ.

### To reproduce the project (without Docker)

1) Clone this repo in your local machine with the command:

    ```bash
    git clone https://github.com/yatesmac/car_price_prediction.git
    ```

1) Use the `cd` command to navigate to the main directory of the project `car_price_prediction`

1) Create a virtual environment and activate it:

    ```bash
    python -m venv env
    source env/bin/activate
    ```

1) Install dependancies

    ```bash
    pip install -r requirements.txt
    ```

1) Use the `cd` to navigate to the `src` directory and run the data preparation in the `data` directory, and model training scripts in the `models` directory:

    ```bash
    cd src/data
    python split_csv.py
    
    cd ../models
    python ann.py
    python rf.py
    python xgb.py
    ```

1) Start the flask application with `gunicorn`:

    ```bash
    cd ../app
    gunicorn --bind 127.0.0.1:9696 predict:app
    ```

    Note: If the above command fails to start up a server, try running gunicorn as a python module:

    ```bash
    python -m gunicorn --bind 127.0.0.1:9696 predict:app
    ```

1) Run the test script:

    ```bash
    python test.py
    ```

Note: You may the comments to edit `test.py` to enable sampling of random rows, and see more results.

## Containerization

### To reproduce the project (with Docker)

1) Clone the repository and navigate to the main project folder (Steps 1 and 2 above).

1) Build the Docker image from Dockerfile

    ```bash
    docker build -t car-prediction .
    ```

1) Run the Docker container from the created image

    ```bash
    docker run -it --rm -p 9696:9696 car-prediction
    ```

1) Test the model. Navigate to project folder and run the test script from another terminal:

    ```bash
    cd src/app
    python test.py
    ```

## About the Dataset

The dataset is a real-world car pricing data from Auto Scout for analysis and price prediction. his dataset provides comprehensive details on used car listings, including vehicle specifications, features, pricing, and more. It's valuable for analyzing car prices, trends, and customer preferences in the automotive market.


The dataset can be accessed on [Kaggle](https://www.kaggle.com/datasets/yaaryiitturan/auto-scout-car-price/data). A copy of the dataset has been added to the repo (in the *data/external* directory).

### Column Descriptions

|Attribute |Description |
|--------- |----------- |
|Comfort_Convenience |Comfort and convenience features, such as 'Air conditioning', 'Leather steering wheel', 'Cruise control', and more. |
|Entertainment_Media |Media features available in the vehicle, including 'Bluetooth', 'MP3', 'Radio', etc. |
|Extras |Additional features like 'Alloy wheels', 'Sport suspension', etc. |
|Safety_Security | Safety features like 'ABS', 'Airbags', 'Electronic stability control', 'Isofix', etc. |
|age |Age of the car (calculated based on the model year). |
|Previous_Owners |The number of previous owners the car has had. |
|hp_kW |Engine power in kilowatts (kW), indicating the performance capacity of the engine. |
|Inspection_new |Indicates whether the car has recently undergone an inspection (1 for yes, 0 for no). |
|Paint_Type |The type of paint on the car, such as 'Metallic', 'Matte', etc. |
|Upholstery_type |The material used for the interior upholstery, such as 'Cloth', 'Leather', etc.|
|Gearing_Type | The type of transmission the car uses, either 'Automatic' or 'Manual'. |
|Displacement_cc |The engine displacement in cubic centimeters (cc), indicating the size of the engine.|
|Weight_kg |The total weight of the vehicle in kilograms. |
|Drive_chain |The type of drivetrain, indicating whether it's 'Front' or 'Rear' wheel drive. |
|cons_comb |The combined fuel consumption in liters per 100 kilometers. |
|make_model |The brand and model of the vehicle (e.g., 'Audi A1'). |
|body_type |The body style of the vehicle, such as Sedan, Compact, or Station Wagon. |
|price |The listed price of the car in currency. |
|vat |Indicates the VAT status for the vehicle's price (e.g., VAT deductible, Price negotiable). |
|km |The total mileage (in kilometers) of the vehicle, indicating its usage. |
|Type |Condition of the vehicle, whether it's 'Used' or 'New'. |
|Fuel |Type of fuel the vehicle uses, such as 'Diesel', 'Benzine', etc. |
|Gears |The number of gears in the vehicle's transmission. |

### Key Features

- Vehicle Specifications: Covers details like make, model, body type, fuel type, and more.
- Comfort & Safety Features: Includes information on air conditioning, safety features, and other convenience options.
- Performance Metrics: Provides data on mileage, engine power, weight, and fuel consumption.
- Pricing Information: Insights into vehicle pricing, VAT status, and other cost-related details.

### Target Variable

The target variable in the dataset is price. The goal is to model car prices based on the above key features (like mileage, fuel type, and performance).

## Results and Evaluation

Linear Regression, Decision Tree, Random Forest, Gradient Boosting, XGboost models were trained on the dataset. Parameter tuning was done for  the best performing models. Random Forest (RF) and XGBoost (XGB) had similar performance. An Artificial Neural Network (ANN) was also trained for comparison.

Thus RF,XGB and ANN were loaded into a webservice using Flask and deployed using a docker container to compare their performance. The Random Forest model found to be the most performant model.

## Acknowledgements

This repository is a project carried out as part of the online course [*Machine Learning Zoomcamp*](https://github.com/DataTalksClub/machine-learning-zoomcamp) instructed by *Alexey Grigorev* and his team from [*DataTalks.Club*](https://datatalks.club/).
