import os
import pandas as pd
import pickle
from flask import Flask, request


# load model
model = pickle.load(
    open('model/model_tch.pkl', 'rb'))

# instanciate flask
app = Flask(__name__)


# 'POST' send data (y_train), 'GET' just get data
@app.route('/predict', methods=['POST'])
def predict():
    test_json = request.get_json()

    # collect data
    if test_json:
        if isinstance(test_json, dict):  # unique value
            df_raw = pd.DataFrame(test_json, index=[0])
        else:
            df_raw = pd.DataFrame(test_json, columns=test_json[0].keys())

    # prediction
    pred = model.predict(df_raw)
    df_raw['prediction'] = pred

    return df_raw.to_json(orient='records')


# start flask
port = os.environ.get('PORT', 6000)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)
