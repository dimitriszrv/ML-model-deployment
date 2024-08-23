from flask import Flask, request 
import pickle 
import numpy as np
from sklearn.metrics import euclidean_distances


local_classifier = pickle.load(open('classifier.pickle','rb'))
local_scaler = pickle.load(open('sc.pickle','rb'))


app = Flask(__name__)

@app.route('/model', methods=['POST'])

def get_data():
    request_data = request.get_json(force=True)
    age = request_data['age']
    salary = request_data['salary']

    # predict-> customer would buy if: age = 40 with salary = 45000 would buy
    prediction = local_classifier.predict(local_scaler.transform(np.array([[age,salary]])))

    # predict the probability -> customer would buy if: age = 40 with salary = 45000 would buy
    prediction_proba = local_classifier.predict_proba(local_scaler.transform(np.array([[age,salary]])))[:,1]
    
    return 'The prediction is {}'.format(prediction) + ' -> with probability: {}'.format(prediction_proba)

    
if __name__ == "__main__":
    app.run(port=8010,debug=True)