import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

class InvalidUsage(Exception):
    """SOURCE: https://flask.palletsprojects.com/en/1.1.x/patterns/apierrors/"""
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv

@app.errorhandler(InvalidUsage)
@app.route('/help', methods=['GET'])
def show_help():
    """
    EXAMPLE REQUEST:
    curl \
      --header "Content-Type: application/json" \
      --request GET \
      http://localhost:5000/help
    """
    return """<!doctype html>
            <html>
            <head>
                <title>Help page</title>
            </head>
            <body>
            <h1>Example cURL POST Request</h1>
            <code>
    curl \
      --header "Content-Type: application/json" \
      --request POST \
      --data '{"event__offer received": 1, "event__offer viewed": 0, "income": 43000, "age_by_decade": 7, "difficulty": 0,
      "duration": 4, "email": 1, "mobile": 1, "social": 0, "web": 1, "offer_type__bogo": 0, "offer_type__discount": 0,
      "offer_type__informational": 1, "amount": 0.111836391, "days_as_customer": 1338, "became_member_dayofweek": 2,
      "became_member_month": 4, "became_member_year": 2017}'   \
      http://localhost:5000/predict
            </code>
            <h1>Example JSON Record</h1>
            <pre>
    {
        "event__offer received": 1,
        "event__offer viewed": 0,
        "income": 43000,
        "age_by_decade": 7,
        "difficulty": 0,
        "duration": 4,
        "email": 1,
        "mobile": 1,
        "social": 0,
        "web": 1,
        "offer_type__bogo": 0,
        "offer_type__discount": 0,
        "offer_type__informational": 1,
        "amount": 0.111836391,
        "days_as_customer": 1338,
        "became_member_dayofweek": 2,
        "became_member_month": 4,
        "became_member_year": 2017
    }
            </pre>
            </body>
            </html>
            """

@app.errorhandler(InvalidUsage)
@app.route('/predict', methods=['POST'])
def predict():
    """
    EXAMPLE API POST REQUEST:
    curl \
      --header "Content-Type: application/json" \
      --request POST \
      --data '{"event__offer received": 1, "event__offer viewed": 0, "income": 43000, "age_by_decade": 7, "difficulty": 0,
      "duration": 4, "email": 1, "mobile": 1, "social": 0, "web": 1, "offer_type__bogo": 0, "offer_type__discount": 0,
      "offer_type__informational": 1, "amount": 0.111836391, "days_as_customer": 1338, "became_member_dayofweek": 2,
      "became_member_month": 4, "became_member_year": 2017}'   \
      http://localhost:5000/predict
    """
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    data = request.json
    try:
        return jsonify({
        'prediction': int(model.predict(np.array(list(data.values())).reshape(1, -1))[0])
        })
    except:
        raise
if __name__ == '__main__':
    app.run(debug=False)

