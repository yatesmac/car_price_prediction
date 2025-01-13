
# Car Price Prediction

![Dataset Cover](/img/dataset_cover.webp)

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

## About the Dataset

The dataset is a real-world car pricing data from Auto Scout for analysis and price prediction. his dataset provides comprehensive details on used car listings, including vehicle specifications, features, pricing, and more. It's valuable for analyzing car prices, trends, and customer preferences in the automotive market.


The dataset can be accessed on [Kaggle](https://www.kaggle.com/datasets/yaaryiitturan/auto-scout-car-price/data). A copy of the dataset has been added to the repo (in the *data/external* directory).

### Column Descriptions

|Attribute |Description |
|----------|------------|
|make_model |The brand and model of the vehicle (e.g., 'Audi A1'). |
|body_type |The body style of the vehicle, such as Sedan, Compact, or Station Wagon. |
|price |The listed price of the car in currency. |
|vat |Indicates the VAT status for the vehicle's price (e.g., VAT deductible, Price negotiable). |
|km |The total mileage (in kilometers) of the vehicle, indicating its usage. |
|Type |Condition of the vehicle, whether it's 'Used' or 'New'. |
|Fuel |Type of fuel the vehicle uses, such as 'Diesel', 'Benzine', etc. |
|Gears |The number of gears in the vehicle's transmission. |
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

### Key Features

- Vehicle Specifications: Covers details like make, model, body type, fuel type, and more.
- Comfort & Safety Features: Includes information on air conditioning, safety features, and other convenience options.
- Performance Metrics: Provides data on mileage, engine power, weight, and fuel consumption.
- Pricing Information: Insights into vehicle pricing, VAT status, and other cost-related details.

### Target Variable

The target variable in the dataset is price. The goal is to model car prices based on the above key features (like mileage, fuel type, and performance).

## Project Structure

```bash
.
├── Dockerfile
├── README.md
├── data                                : Directory containing 
│   ├── external
│   │   └── final_scout_not_dummy.csv   : CSV dataset downloaded from Kaggle
│   ├── test                            :
│   │   ├── df_test.csv
│   │   └── random_rows
│   │       ├── row_3.csv
│   │       └── row_59.json
│   └── train
│       ├── df_full_train.csv
│       ├── df_train.csv
│       └── df_val.csv
├── img
│   └── dataset_cover.webp
├── logs
│   └── debug.log
├── models
│   ├── ann_v1.h5
│   ├── ann_v1.keras
│   ├── rf_v1.pkl
│   └── xgb_v1.pkl
├── notebooks
│   ├── 1_data_exploration.ipynb
│   ├── 2_data_preparation.ipynb
│   ├── 3_skl_xgb_models.ipynb
│   └── 4_ann_model.ipynb
├── pyproject.toml
├── requirements.txt
└── src
    ├── app
    │   ├── predict.py
    │   ├── sample.py
    │   ├── test.py
    │   └── testing.ipynb
    ├── data
    │   ├── data_preparation.py
    │   ├── flattencolumns.py
    │   └── split_csv.py
    └── models
        ├── ann.py
        ├── rf.py
        └── xgb.py
        
```

The function of scripts in the `src` directory:

```bash
    src
    ├── data_prep.py  <-- Prepare data before training
    ├── predict.py    <-- Flask application
    ├── test.py       <-- Test Flask application
    └── train.py      <-- Train Logistic Regression model and save to Pickle file
```

## Project Setup

### To reproduce the project (without Docker)

1) Clone this repo in your local machine with the command:

    ```bash
    git clone https://github.com/
    ```

1) Use the `cd` command to navigate to the main directory of the project `student_evaluation`

1) Create a virtual environment and activate it:

    ```bash
    python -m venv env
    source env/bin/activate
    ```

1) Install dependancies

    ```bash
    pip install -r requirements.txt
    ```

1) Use the `cd` to navigate to the `src` directory and run the data preparation, and model training scripts:

    ```bash
    cd src
    python data_prep.py
    python train.python
    ```

1) Start the flask application with `gunicorn`:

    ```bash
    gunicorn --bind 127.0.0.1:9696 predict:app
    ```

1) Run the test script:

    ```bash
    python test.py
    ```

## Containerization

### To reproduce the project (with Docker)

1) Clone the repository and navigate to the main project folder (Steps 1 and 2 above).

1) Build the Docker image from Dockerfile

    ```bash
    docker build -t student-evaluation .
    ```

1) Run the Docker container from the created image

    ```bash
    docker run -it --rm -p 9696:9696 student-evaluation
    ```

1) Test the model. Navigate to project folder and run the test script from another terminal:

    ```bash
    cd src
    python test.py
    ```

## Results and Evaluation

Five models: Logistic Regression, Decision Tree, Random Rorest, Gradient Boosting and XGboost were trained on the dataset. Parameter tuning was done for each model and Logistic Regression was found to be the best performing model.

The Logistic Regression model was thus train on the training and validation datasets, then saved. Afterwards, it was loaded into a webservice using Flask and deployed using a docker container.

## Acknowledgements

This repository is a project carried out as part of the online course [*Machine Learning Zoomcamp*](https://github.com/DataTalksClub/machine-learning-zoomcamp) instructed by *Alexey Grigorev* and his team from [*DataTalks.Club*](https://datatalks.club/).