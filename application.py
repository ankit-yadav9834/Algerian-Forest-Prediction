import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify, render_template


application = Flask(__name__)
app = application

## import ridge regression and standardscaler pickle

ridge_model = pickle.load(open('model/ridgeReg.pkl', 'rb'))
Standard_scaler = pickle.load(open('model/scaler.pkl', 'rb'))



@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == "POST":
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        DC = float(request.form.get('DC'))
        ISI = float(request.form.get('ISI'))
        BUI = float(request.form.get('BUI'))
        
        new_data = Standard_scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, DC, ISI, BUI]])
        result = ridge_model.predict(new_data)

        return render_template('home.html', results=round(result[0], 2))

    else:
        return render_template('home.html', results="")



if __name__ == '__main__':
    app.run(host= "0.0.0.0")