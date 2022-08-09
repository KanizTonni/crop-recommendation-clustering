# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 11:36:48 2022

@author: siddhardhan
"""

import json
import pickle

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class model_input(BaseModel):
    
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float
    

# loading the saved model
Cluster_pickle_in = open("KMeansCluster.pkl","rb")
RFclassifier=pickle.load(Cluster_pickle_in)

@app.post('/crop-predict')
def diabetes_pred(input_parameters : model_input):
    
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)
    
    N = input_dictionary['N']
    P = input_dictionary['P']
    K = input_dictionary['K']
    temperature = input_dictionary['temperature']
    humidity = input_dictionary['humidity']
    ph = input_dictionary['ph']
    rainfall = input_dictionary['rainfall']

    prediction = RFclassifier.predict([[N,P,K,temperature,humidity,ph,rainfall]])
    print(prediction)
    return str(prediction[0])

