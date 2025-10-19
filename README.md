## Stock Price Prediction using Linear Regression
## Overview

This project implements a stock price prediction system using a Linear Regression model. It leverages historical stock data from Yahoo Finance, applies feature engineering to create technical indicators, trains a predictive model, evaluates its performance, and forecasts future stock prices for the next 7 days. The code is designed to be modular, reusable, and easy to extend for further experimentation.

## Features

### Data Retrieval:
Downloads historical stock data using the yfinance library for a user-specified stock ticker and time period.
### Feature Engineering: 
Computes technical indicators such as Moving Averages (MA5, MA20), Relative Strength Index (RSI), Daily Returns, and temporal features like Day of Week and Day of Month.
### Model Training: 
Trains a Linear Regression model with standardized features, using a train-test split to ensure robust evaluation.
### Performance Evaluation: 
Calculates R² and RMSE metrics for both training and test sets to assess model accuracy.
### Visualization: 
Generates plots to compare actual vs. predicted prices and a scatter plot to visualize prediction accuracy.
### Forecasting: 
Predicts stock prices for the next 7 days by iteratively updating features based on previous predictions.
### Model Persistence: 
Saves the trained model and scaler for future use and allows loading for predictions without retraining.

## Requirements

To run this project, ensure you have the following Python packages installed:

yfinance: For fetching stock data.  
numpy: For numerical computations.  
pandas: For data manipulation and analysis.  
scikit-learn: For machine learning model and preprocessing.  
matplotlib: For plotting results.  
joblib: For saving and loading the model.  

Install the dependencies using:
pip install yfinance numpy pandas scikit-learn matplotlib joblib

## Usage

Run the Script:
Execute the script using Python:python stock_price_prediction.py

Input the stock ticker (e.g., AAPL for Apple Inc.) and the number of months of historical data (e.g., 12).


## Output:

The script fetches and saves the stock data as a CSV file (e.g., aapl_stock_data_36.csv).  
It trains a Linear Regression model and outputs performance metrics (R² and RMSE for training and test sets).  
A plot (Regression_Prediction.png) is generated, showing actual vs. predicted prices and a scatter plot of predictions.  
The script predicts and displays the stock prices for the next 7 days.  
The trained model and scaler are saved as linear_regression_model.pkl.  


## Customization:

Modify the features list in the run() function to experiment with different input features.  
Adjust the days parameter in predict_next_days() to forecast a different number of days.  
Change the plotting style or metrics in plot_results() for alternative visualizations.  

## Code Structure

fetch_stock_data(symbol, period): Downloads historical stock data for the specified ticker and period.  
calculate_rsi(prices, periods): Computes the Relative Strength Index (RSI) for the given price series.  
prepare_features(df): Adds technical indicators and temporal features to the dataset.  
train_model(df, features): Trains the Linear Regression model and returns evaluation metrics and test data.  
plot_results(test_results, file_name): Visualizes actual vs. predicted prices and a scatter plot.  
predict_next_days(model, scaler, df, features, days): Forecasts future stock prices iteratively.  
save_model(model, scaler, filename) and load_model(filename): Handle model persistence.  
run(): Orchestrates the entire workflow, from data fetching to prediction.  

## Example Output
Fetching AAPL data for last 36 months...  
Data saved to aapl_stock_data_36.csv  
Model saved to linear_regression_model.pkl  

Model Performance:  
train_r2: 0.9994  
test_r2: 0.9952  
train_rmse: 0.80  
test_rmse: 1.35  
Charts saved in Regression_Prediction.png  

Next 7 days predicted prices:  
Day1 : 251.775  
Day2 : 250.736  
Day3 : 250.461  
Day4 : 250.389  
Day5 : 250.370  
Day6 : 250.365  
Day7 : 250.363  

## Limitations

The model assumes linear relationships between features and stock prices, which may not capture complex market dynamics.
Predictions are based on historical data and technical indicators, which do not account for external factors like news or economic events.
The iterative forecasting method may accumulate errors over longer prediction horizons.

## Future Improvements

Incorporate additional technical indicators (e.g., Bollinger Bands, MACD) or fundamental data (e.g., earnings reports).
Experiment with advanced models like LSTM or Random Forest for improved accuracy.
Add cross-validation to enhance model robustness.
Include real-time data fetching for more up-to-date predictions.

## License
This project is licensed under the MIT License. Feel free to use, modify, and distribute the code as needed.
## Author
### Zahra Shakeri
