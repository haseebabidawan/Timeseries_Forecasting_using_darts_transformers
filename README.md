# Temporal Fusion Transformer (TFT) Demand Forecasting

This project implements demand forecasting using the Temporal Fusion Transformer (TFT) model. The main objective is to predict the quantity of a product (or other demand metric) based on historical sales data and external covariates.
![Time Series Forcasting](https://2.bp.blogspot.com/-s4AbzDcZNaY/XBTGDoa9iRI/AAAAAAAADko/CBtR7qUDHqYnh3zd1-DNGkwuQExTmgJSwCLcBGAs/s640/TimeSeries.jpg)
## Overview

This project consists of two primary functionalities:
1. **Model Training**: The TFT model is trained on historical sales data to learn patterns and relationships.
2. **Demand Prediction**: The trained model is used to predict future demand, and the performance is evaluated using metrics such as Mean Absolute Percentage Error (MAPE).

The process involves:
- Preprocessing and preparing the dataset.
- Training a TFT model using time series data.
- Predicting future demand and evaluating the results.
- Visualizing the predictions and actual sales data.
- Saving the final predictions for later use.

## Requirements

Before running the code, ensure that you have the following Python libraries installed:
- `pandas`
- `numpy`
- `matplotlib`
- `sklearn`
- `darts`
- `relativedelta`
- `scipy`
- `joblib`

You can install the required packages by running:

```bash
pip install pandas numpy matplotlib scikit-learn darts scipy joblib

```
## Data Format

The dataset should have the following columns:

- `YEAR_QUARTER`: Year and quarter in the format `YYYY Qn` (e.g., `2023 Q1`).
- `MODEL`: The product or item for which demand is being forecasted.
- `CUSTOMER`: The customer identifier (if applicable).
- `QUANTITY`: Historical sales data (target variable).

Additionally, the dataset may include external covariates, such as promotions or other features, to improve forecasting accuracy.

## Structure

### Main Script (`main.py`)

1. **Data Preparation**:
    - Load and merge datasets, fill missing values, and transform the data.
    - Create features like `QUARTER`, `YEAR`, and `QUARTER-YEAR` for easier processing.

2. **Training**:
    - Split the data into training and validation sets based on a specified `split_date`.
    - Apply `MinMaxScaler` to normalize the data and train the TFT model using the `train_model()` function.

3. **Prediction**:
    - Use the trained model to predict future demand for each product model.
    - Perform backtesting and calculate forecasting accuracy using MAPE.

4. **Visualization**:
    - Plot actual vs. predicted demand for visual comparison.
    - Save the final predictions in a CSV file for further analysis.

### Functions

- `train_model()`: Trains the TFT model on the transformed data.
- `predict_model()`: Predicts future demand using the trained model.
- `perform_backtest()`: Evaluates the model’s performance on past data.
- `eval_backtest()`: Computes metrics such as MAPE to evaluate prediction accuracy.

## Steps to Run the Code

### 1. Set Up the Parameters

You can configure the behavior of the script by adjusting the following variables in the script:

- `TRAIN`: Set to `'Yes'` to train the model or `'No'` if you only want to make predictions.
- `PREDICT`: Set to `'Yes'` to generate predictions or `'No'` if you only want to train the model.
- `Algo`: The algorithm used for training (e.g., `'TFT'`).
- `outlier`: Set to `'Yes'` or `'No'` depending on whether you want to consider outliers during model training.
- `var_cti`: Set to `'Yes'` or `'No'` to include external covariates during training.
- `Prediction_Range`: The time period for which predictions are generated (e.g., `4` for 4 quarters).
- `current_date`: The current date of execution to track when the predictions are generated.

### 2. Data Preparation

Ensure your dataset is available and has the required columns. The `final_df` should be a DataFrame with the columns mentioned above (`YEAR_QUARTER`, `MODEL`, `CUSTOMER`, `QUANTITY`, etc.).

### 3. Running the Script

To execute the script, simply run the following command:

```bash
python main.py
```
### 4. Output
The predictions and evaluation metrics are saved in the prediction and metrics_results DataFrames. These will be concatenated and can be exported to CSV or analyzed further. For each model, the predicted demand will be visualized on a plot, showing both the actual demand and the predicted values. The MAPE score is displayed on the plot to indicate the accuracy of the model's predictions.

```bash
Model Training : Starting Training for Algo: 'TFT' with Outlier and with Covariates
Demand Prediction : Starting Demand Prediction Using Algo: 'TFT' with Outlier and with Covariates
Prediction Using 'TFT' Completed
MAPE: 5.23% for the model XYZ
Training Using 'TFT' Completed
```
### Conclusion
This project demonstrates how to use the Temporal Fusion Transformer (TFT) model for time series forecasting. By training the model on historical demand data and external covariates, it can predict future demand with a high degree of accuracy. The final predictions and model performance metrics are saved and visualized for further analysis. Feel free to modify the code or add more functionality, such as hyperparameter optimization or other evaluation metrics, to improve the performance of the forecasting model.

### License
This project is licensed under the MIT License - see the LICENSE file for details.

### Customization Tips

* Adjust the **data columns** section to match your exact data structure if needed.
* Update any specific **dependencies** based on the libraries you’re using.
* You can add more sections like **Known Issues**, **Troubleshooting**, or **Contact Info** based on your project needs.
