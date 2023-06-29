from flask import Flask, render_template,redirect,request
import pickle
import pandas as pd
import numpy as np
app = Flask(__name__)


app=Flask(__name__, template_folder='./templates', static_folder='./static')


rfr = pickle.load(open('PopulationGrowth.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict():
    features=[float(x) for x in list(request.form.values())]
    final_features =np.array(features).reshape((1,2))
    # final_features = final_features.reshape(-1, 1)

    result = rfr.predict(final_features)
    return render_template('result.html',result=result[0])

if __name__=='__main__':
    app.run(debug=True)
