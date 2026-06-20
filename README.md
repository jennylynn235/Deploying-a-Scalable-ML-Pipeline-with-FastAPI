# Deploying a Scalable Machine Learning Pipeline Using FastAPI

## Overview

This project demonstrates the end-to-end development and deployment of a machine learning pipeline that predicts whether an individual's annual income exceeds **$50K** using the U.S. Census Income dataset. The project includes data preprocessing, model training, model evaluation, unit testing, a RESTful API built with FastAPI, and continuous integration using GitHub Actions. 

## Project Highlights
- Built an end-to-end machine learning pipeline for Census Income classification
-Developed a RESTful API using FastAPI for real-time inference
- Implemented six unit tests with Pytest
- Configured GitHub Actions for automated testing and continuous intergration
- Documented the model using a Model Card

## Features

* Data preprocessing and feature engineering
* Random Forest classification model
* Model serialization using pickle
* Slice-based model performance evaluation
* Six unit tests using Pytest
* RESTful API built with FastAPI
* Local API testing using the Python Requests library
* Continuous Integration with GitHub Actions
* Model Card documentation

## Environment Setup

This project can be configured using either Conda or pip. For this project, I used pip to manage the development environment.

### Option 1: Conda

Create the environment using the provided `environment.yml` file.

### Option 2: pip

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

## Running the Project

### Train the Model

```bash
python train_model.py
```

### Start the FastAPI Application

```bash
uvicorn main:app --reload
```

### Test the API Locally

```bash
python local_api.py
```

## Project Structure

```
.
├── data/
├── ml/
├── model/
├── screenshots/
├── local_api.py
├── main.py
├── model_card.md
├── slice_output.txt
├── test_ml.py
├── train_model.py
├── requirements.txt
└── README.md
```

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* FastAPI
* Pydantic
* Requests
* Pytest
* GitHub Actions


## GitHub Repository:

https://github.com/jennylynn235/Deploying-a-Scalable-ML-Pipeline-with-FastAPI

**Completed as part of the Udacity Machine Learning DevOps Engineer Nanodegree in partnership with Western Governors University (WGU).**