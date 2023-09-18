from flask import Flask,render_template,request,app,jsonify,url_for
import numpy as np
import pickle
import pandas as pd

app=Flask(__name__)

#load  model
model=pickle.load(open("regmodel2.pkl","rb"))


#define route

@app.route('/')
def home():

    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])


@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json()['data']  # Accédez au sous-dictionnaire 'data'
    print(data)

    # Créez un tableau NumPy à partir des valeurs
    data_array = np.array(list(data.values())).reshape(1, -1)
    
    # Effectuez la prédiction
    output = model.predict(data_array)
    print(output[0])

    return jsonify(float(output[0]))


if __name__=="__main__":
    app.run(debug=True)






