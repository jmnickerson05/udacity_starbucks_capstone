import pickle
import numpy as np
# import warnings
# warnings.filterwarnings("ignore")
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    data = request.json
    return jsonify({
        'prediction': int(model.predict(np.array(list(data.values())).reshape(1, -1))[0])
    })

if __name__ == '__main__':
    app.run(debug=True)

# curl \
#   --header "Content-Type: application/json" \
#   --request POST \
#   --data '{"event__offer received": 1, "event__offer viewed": 0, "income": 43000, "age_by_decade": 7, "difficulty": 0, "duration": 4, "email": 1, "mobile": 1, "social": 0, "web": 1, "offer_type__bogo": 0, "offer_type__discount": 0, "offer_type__informational": 1, "amount": 0.111836391, "days_as_customer": 1338, "became_member_dayofweek": 2, "became_member_month": 4, "became_member_year": 2017}'   \
#   http://localhost:5000/predict
