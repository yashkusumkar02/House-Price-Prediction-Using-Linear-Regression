# House-Price-Prediction-Using-Linear-Regression

## Overview

This project is a house price prediction case study that utilizes Linear Regression to model the relationship between house features and their selling prices. The dataset contains information such as the number of bedrooms, bathrooms, square footage, lot size, and other features. The goal of the project is to predict the price of a house based on its features using a linear regression model.

The case study includes detailed steps such as Exploratory Data Analysis (EDA), feature engineering, data preprocessing, and model training. The project explores different techniques for fitting the Ordinary Least Squares (OLS) model and includes a comparative analysis of the accuracy and performance of the model.

## Problem Statement

Predicting house prices is a classic regression problem in data science. Given the large number of features that influence the price of a house, the objective is to develop a machine learning model that can accurately predict the house prices based on these features.

## Dataset

The dataset includes the following features:

- id: Unique identifier of the house
- bedrooms: Number of bedrooms
- bathrooms: Number of bathrooms
- sqft_living: Square footage of living area
- sqft_lot: Size of the lot in square feet
- floors: Number of floors in the house
- zipcode: Zip code of the location
- year: Year the house was built
- month: Month of the transaction
- day: Day of the transaction


## Target Variable:

- price: The price at which the house was sold.

## Project Workflow

### 1. Exploratory Data Analysis (EDA)

The project starts with an in-depth EDA, which includes:

- Visualizing relationships between features and the target variable.
- Checking for missing values and understanding the distribution of the data.
- Identifying correlations between numerical variables and the target variable using correlation matrices.

### 2. Data Preprocessing

Before model training, the data is cleaned and preprocessed:

- Handling missing values (if any) to ensure a complete dataset.
- Converting categorical variables (such as zipcode) into dummy variables for regression.
- Standardizing numerical features for better model performance.
- Splitting the data into training and testing sets.

### 3. Model Building Using Linear Regression

The project fits a Linear Regression model to the data using two approaches:

- Scikit-Learn: A basic linear regression model using the LinearRegression() class.
- Statsmodels OLS: A more detailed Ordinary Least Squares (OLS) regression model using statsmodels to obtain detailed statistical metrics such as p-values, confidence intervals, and R-squared values.

### 4. OLS Regression Diagnostics

The OLS model provides insights into:

- R-squared: Explains how well the independent variables explain the variation in the dependent variable (house prices).
- P-values: Indicates the significance of each feature in predicting house prices.
- Coefficients: Show the weight of each feature in the linear equation.


### 5. Model Evaluation

After fitting the model, the performance is evaluated using:

- Mean Squared Error (MSE): Measures the average squared difference between actual and predicted values.
- R-squared score: Quantifies the goodness of fit of the model.

## Key Features of the Project

* Model Implementation: The project implements multiple ways to fit a Linear Regression model, including both scikit-learn and statsmodels.
* Data Visualization: Includes visualization techniques such as pair plots, correlation heatmaps, and regression plots.
* Feature Engineering: Uses dummy variables for categorical features such as zipcode and creates derived features such as year, month, and day from the transaction date.
* Detailed Statistical Summary: With statsmodels, the OLS regression model provides detailed insights into the significance of each predictor.
* Customizability: The project is easily extendable to incorporate new features or test different machine learning models.


## Tools and Technologies Used

- Python: The entire project is written in Python.
- Pandas: For data manipulation and preprocessing.
- NumPy: For numerical computations.
- Matplotlib & Seaborn: For data visualization.
- Scikit-Learn: For model building and evaluation.
- Statsmodels: For building OLS regression and obtaining detailed statistics.
- Jupyter Notebook: The project is organized as an interactive notebook.
