# load the libraries
import pandas as pd
import joblib
import numpy as np
from waitress import serve
from flask import Flask, redirect, url_for, request, render_template

# suppress all warnings (ignore unnecessary warnings msgs)
import warnings
warnings.filterwarnings("ignore")


# define the flask and template directory 
app = Flask(__name__,template_folder='templates')

# load the model at the start
filename = "RF.model" #filename
loaded_model = joblib.load(filename)


# serve the index 
@app.route("/")
def index():
    return render_template('form.html')

# handle the form action
@app.route("/result", methods=["POST"])
def prediction_result():
    
    # receiving the POST data from the client 
    # (Form submitted by the client/user)
    age = request.form.get('age')
    anaemia = request.form.get('anaemia')
    creatinine_phosphokinase = request.form.get('creatinine_phosphokinase')
    diabetes = request.form.get('diabetes')
    ejection_fraction = request.form.get('ejection_fraction')
    high_blood_pressure = request.form.get('high_blood_pressure')
    platelets = request.form.get('platelets')
    serum_creatinine = request.form.get('serum_creatinine')
    serum_sodium = request.form.get('serum_sodium')
    sex = request.form.get('sex')
    smoking = request.form.get('smoking')
    time = request.form.get('time')
    
    '''
    features = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes','ejection_fraction', 'high_blood_pressure', 'platelets','serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time']
    '''
    # convert new data into numpy array list
    new_data = np.array([[age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, time]])
    
    #predict new_data
    new_data_pred = loaded_model.predict(new_data)

    # convert the prediction from number to word
    if new_data_pred[0] == 1:
        output = "death"
    elif new_data_pred[0] == 0:
        output = "survived"
    else:
        output = "unknown"
        
    return render_template('result.html', age=age, anaemia=anaemia, creatinine_phosphokinase=creatinine_phosphokinase, diabetes=diabetes, ejection_fraction=ejection_fraction, high_blood_pressure=high_blood_pressure, platelets=platelets, serum_creatinine=serum_creatinine, serum_sodium=serum_sodium, sex=sex, smoking=smoking, time=time, output=output)

if __name__ == "__main__":
    '''
     # change the port number, available from 5000-50021 
     (there are 21 port slots, please choose one and post in the chat 
     so that other student can choose the available one)
    '''
    serve(app, host='0.0.0.0', port=5001)


