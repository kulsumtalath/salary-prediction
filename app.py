# -*- coding: utf-8 -*-
"""
Created on Mon May 31 20:34:23 2021

@author: Kulsum
"""

import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle

app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final_features=[np.array(int_features)]
    prediction=model.predict(final_features)
    
    output=round(prediction[0],2)
    return render_template('index1.html',prediction='Employee Salary should be $ {}'.format(output))

if __name__=="__main__":
    app.run(debug=True)