# Starbucks Capstone Challenge

### Introduction

This data set contains simulated data that mimics customer behavior on the Starbucks rewards mobile app.

The deliverable item for this project is a Flask API endpoint that takes a JSON record of variables and returns a binary
classification of 0 for probable failure to convert and 1 for a probable conversion given the features available. 

The core use case for this classification model is to simulate (or model) the real world probability of customer conversion
given a series of known variables.

In the context of an application that calls a prediction endpoint, the important variables that are able to be manipulated 
via the app can be changed to heighten the probability of conversion. E.g. if a customer has received but not viewed an offer
for instance then the app can increase notification frequency of pending offer, show a countdown to offer expiration, etc.. until
the a threshold is crossed and the prediction API indicates a positive value for probability to convert.

## Datasets

The data is contained in three files:

* portfolio.json - containing offer ids and meta data about each offer (duration, type, etc.)
* profile.json - demographic data for each customer
* transcript.json - records for transactions, offers received, offers viewed, and offers completed

Here is the schema and explanation of each variable in the files:

**portfolio.json**
* id (string) - offer id
* offer_type (string) - type of offer ie BOGO, discount, informational
* difficulty (int) - minimum required spend to complete an offer
* reward (int) - reward given for completing an offer
* duration (int) - time for offer to be open, in days
* channels (list of strings)

**profile.json**
* age (int) - age of the customer 
* became_member_on (int) - date when customer created an app account
* gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
* id (str) - customer id
* income (float) - customer's income

**transcript.json**
* event (str) - record description (ie transaction, offer received, offer viewed, etc.)
* person (str) - customer id
* time (int) - time in hours since start of test. The data begins at time t=0
* value - (dict of strings) - either an offer id or transaction amount depending on the record

## Core Libraries
####Machine Learning:
* Scikit-Learn 
* XGBoost
####API Endpoint
* Flask
####Analysis, exploration, and model evaluation:
* JupyterLab
* Pandas
* Statsmodel
 

## Setup
    python3 -m ven env && source env/bin/activate
    pip install -r requirements.txt

## Run Application 
    # Launch Flask App
    python -m app.py
    # Execute example prediction endpoint request using cURL (in a new terminal window/tab)
    bash curl_example.sh
    
