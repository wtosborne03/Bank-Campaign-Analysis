# Bank Market Outcome Analysis

This repository contains a comprehensive analysis of a bank marketing dataset with the aim of predicting campaign success. Using data cleaning techniques and XGBoost, a gradient boosting framework, we have developed a predictive model to assess the likelihood of clients subscribing to a term deposit following a bank's marketing campaign.

## Overview

The project revolves around the dataset that includes client information, their responses to previous marketing campaigns, and the outcome of the current campaign. We perform extensive data cleaning, feature engineering, and apply XGBoost to predict whether a client will subscribe to a term deposit.

## Features

The dataset consists of various features, including:

- **Client Information**: Age, job type, marital status, education level, default history, balance, housing, and loan status.
- **Campaign Data**: Contact method, day and month of the last contact, duration, campaign, pdays, previous, and poutcome.
- **Target Variable**: Whether the client subscribed to the term deposit (y).

## Dependencies

This project requires the following Python libraries:
- NumPy
- pandas
- XGBoost
- Seaborn
- Matplotlib
- scikit-learn
- 
## Model Inference

The trained model can be used to make predictions on new data, offering a tool for bank marketing teams to strategize more effectively.
![Inference Screenshot](/images/inference_screenshot.png?raw=true "Inference Screenshot Form")


## Credits

https://www.kaggle.com/datasets/ara001/bank-client-attributes-and-marketing-outcomes/data
