import json
import logging
import os
import joblib
import pytest
from prediction_service.prediction import form_response, api_response
import prediction_service

input_data = {
    "incorrect_range": 
    {
        "Pregnancies": 30,
        "Glucose": 500,
        "BloodPressure": 200,
        "SkinThickness": 300,
        "Insulin": 2000,
        "BMI": 200,
        "DiabetesPedigreeFunction": 20,
        "Age": 120
    },

    "correct_range":
    {
        "Pregnancies": 2,
        "Glucose": 120,
        "BloodPressure": 100,
        "SkinThickness": 40,
        "Insulin": 600,
        "BMI": 30,
        "DiabetesPedigreeFunction": 1,
        "Age": 50
    },

    "incorrect_col":
    {
        "Pregnancies": 2,
        "Glucose": 120,
        "Blood Pressure": 100,
        "Skin Thickness": 40,
        "Insulin": 600,
        "BMI": 30,
        "Diabetes Pedigree Function": 1,
        "Age": 50
    }
}

TARGET_range = {
    "min": 0.0,
    "max": 1.0
}

def test_form_response_correct_range(data=input_data["correct_range"]):
    res = form_response(data)
    assert  TARGET_range["min"] <= res <= TARGET_range["max"]

def test_api_response_correct_range(data=input_data["correct_range"]):
    res = api_response(data)
    assert  TARGET_range["min"] <= res["response"] <= TARGET_range["max"]

def test_form_response_incorrect_range(data=input_data["incorrect_range"]):
    with pytest.raises(prediction_service.prediction.NotInRange):
        res = form_response(data)

def test_api_response_incorrect_range(data=input_data["incorrect_range"]):
    res = api_response(data)
    assert res["response"] == prediction_service.prediction.NotInRange().message

def test_api_response_incorrect_col(data=input_data["incorrect_col"]):
    res = api_response(data)
    assert res["response"] == prediction_service.prediction.NotInCols().message