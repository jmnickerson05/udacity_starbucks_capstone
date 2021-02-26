# Starbucks Capstone Challenge

## Project Overview

This data set contains simulated data that mimics customer behavior on the Starbucks rewards mobile app.

The deliverable item for this project is a Flask API endpoint that takes a JSON record of variables and returns a binary
classification of 0 for probable failure to convert and 1 for a probable conversion given the features available. 

The core use case for this classification model is to simulate (or model) the real world probability of customer conversion
given a series of known variables.

In the context of an application that calls a prediction endpoint, the important variables that are able to be manipulated 
via the app can be changed to heighten the probability of conversion. E.g. if a customer has received but not viewed an offer
for instance then the app can increase notification frequency of pending offer, show a countdown to offer expiration, etc.. until
the a threshold is crossed and the prediction API indicates a positive value for probability to convert.

##Problem Statement

The goal here is predicting customer conversion from customer promotions and/or the degree to which promotions or customer attributes affect behaviour.

##Methodology and Data Pre-processing

### Data Issues:
- Multicolinearity - VIF scores:
    - Multiple columns with high VIF scores were removed.
    - NOTE: Originally a RandomForest model was chosen which is affected by multicolinearity, but I ended up switching to using
    a Naive Bayes algorithm (GaussianNB) which is not affected by multicolinearity.
- Class Imbalance Issues - Resampling:
    - ```from sklearn.utils import resample``` was used for this
- Datasets containing multiple entities (offers/transcripts)
    - The data is not well normalized or fully denormalized and contained nested JSON values, so the preparatory data massaging was needed.
 
### Data Cleaning and Feature engineering:
- Engineering new features for age, gender, channels, date-parts, etc..
- Removing columns:a
    - Non features columns such as ID columns need to be removed.

### Visualizations:
- Distributions of variables:
    - Age
    - Income
    - Days as Customer
    - Gender
- ROC curve plot for model evaluation
- Feature Importance plot

### Modeling and Evaluation:
- Comparison of multiple classification algorithms using multiple classification metrics, classification reports, plots, and output dataframes.
- Naive Bayes was ultimately chosen even though it was not the best performing algorithm. 
    - NOTE: Some of the algorithms examined were 100% accuracy leading me to believe that there were overfitting 
    (or that there was a variable that I overlooked that was 100% correlated with the target Y variable).
    I went with Naive Bayes because it was semi-accurate, more resilient to multicolinearity and such, 
    and because it didn't seem to be overfitting.

### Discussion / Reflection
The classification model would likely be of minimal value if deployed in an app - predicting that a given customer will fulfill a offer doesn't make it more likely to happen to or create additional value for the organization.

However, it may be useful for modeling real world customer lifecycles. It possible that this model could be used for marketing purposes. For example, if there are a number of important variables that can be manipulated by the organization or a group of people that can be targeted to increase customer conversion then it may be able to provide actual, measurable value.

One caveat here is that further causal inference needs to be exercised for which this model wouldn't necessarily be the best. A multiple regression model, A/B testing, and statistical inference techniques in general would be much better. Prediction and causality are two separate problem spaces.

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
### Machine Learning:
* Scikit-Learn 
* XGBoost (not used for final model)

### API Endpoint
* Flask

### Analysis, exploration, and model evaluation:
* JupyterLab
* Pandas
* Statsmodel
 
## Repo files:
- Core Python development artifacts?
    - Starbucks_Capstone_notebook.ipynb
    - utils.py - module to contain core Python functions
- PNG for embedded markdown images
- Setup scripts:
    - Requirements.txt
    - Dockerfile
    - setup_env.sh
    - NOTE: The Dockerfile and Bash script are not necessary components, but I chose to leave these in regardless.
- Data directory:
    - JSON data files
    - Starbucks.db SQLite file: 
        - NOTE: This is not pushed with the repo because of Github size restrictions and the fact that binary files are 
        not able to be adequately compressed. Such files cause the repo to be over-sized and bloated over time.
    - Modeling components:
        - model.pkl = pickled Naive Bayes model.
        - grid_search_results.pkl:
            - NOTE: this is not needed for the final model because Naive Bayes doesn't have any parameters to tune,
            but I left it in the repo for posterity.
- Flask app components:
    - app.py - the Flask application
    - curl_example.sh
    - example.json

## Setup
    python3 -m ven env && source env/bin/activate
    pip install -r requirements.txt

## Run Application 
    # Launch Flask App
    python -m app.py
    # Execute example prediction endpoint request using cURL (in a new terminal window/tab)
    bash curl_example.sh
    
## Acknowledgements
NOTE: Credits also present in function Docstrings

Examples snippets taken/adapted from the following sources:
   * https://inblog.in/Feature-Importance-in-Naive-Bayes-Classifiers-5qob5d5sFW
   * https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
   * https://stackoverflow.com/questions/29034928/pandas-convert-a-column-of-list-to-dummies
   * https://towardsdatascience.com/methods-for-dealing-with-imbalanced-data-5b761be45a18
   * https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/