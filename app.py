from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
model = pickle.load(open('US_prediction.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    try:
        if request.method == 'POST':
            GRE = float(request.form['GRE'])
            TOEFL=float(request.form['TOEFL'])
            UR=float(request.form['UR'])
            SOP=float(request.form['SOP'])
            LOR=float(request.form['LOR'])
            GPA=float(request.form['GPA'])
            RES = float(request.form['RES'])
            prediction=model.predict([[GRE,TOEFL,UR,SOP,LOR,GPA,RES]])
            output=prediction[0]
            if (SOP>5 or GRE>340 or LOR>5 or GPA>10 or UR>5 or TOEFL>120):
                return render_template('index.html', prediction_text="ENTER CORRECT VALUES")
            else:
                if output==1:
                    return render_template('index.html',prediction_text="MOST PROBABLY ADMIT")
                else:
                    return render_template('index.html',prediction_text="MOST PROBABLY REJECT")
        else:
            return render_template('index.html')
    except:
        return render_template('index.html', prediction_text="Pls enter valid numbers as score")
if __name__=="__main__":
    app.run(debug=True)