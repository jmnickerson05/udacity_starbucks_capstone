import pandas as pd
import numpy as np
import math
import json
import functools
import sqlite3
import seaborn as sns
from IPython.core.magic import (Magics, magics_class, line_magic, cell_magic)
from IPython import get_ipython
from datetime import datetime
from sklearn.metrics import make_scorer, f1_score, fbeta_score, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.utils import resample
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score, fbeta_score, accuracy_score
import pickle
from time import time
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

def none_on_exception(fn):
	@functools.wraps(fn)
	def inner(*args, **kwargs):
		try:
			return fn(*args, **kwargs)
		except Exception:
			return None
	return inner

adapt_json = lambda data: (json.dumps(data, sort_keys=True)).encode()
convert_json = lambda blob: json.loads(blob.decode())
sqlite3.register_adapter(dict, adapt_json)
sqlite3.register_adapter(list, adapt_json)
sqlite3.register_adapter(tuple, adapt_json)
sqlite3.register_converter('JSON', convert_json)
conn = sqlite3.connect('data/starbucks.db', detect_types=sqlite3.PARSE_DECLTYPES)

@magics_class
class SqlMagic(Magics):
    @cell_magic
    def sql(self, line, cell):
        return conn.execute(cell).fetchall()

    @cell_magic
    def to_df(self, line, cell):
        return pd.read_sql(cell, conn)
    
get_ipython().register_magics(SqlMagic)    
sql = lambda x: conn.execute(x).fetchall()
to_df = lambda x: pd.read_sql(x, conn)
conv_date = lambda a: datetime.strptime(str(int(a)), '%Y%m%d').strftime('%m/%d/%Y') if not np.isnan(a) else np.NaN

# read in the json files
def load_json_file_to_table(fname):
    print(f'Loading data from {fname}...')
    fname = fname.replace('.json','')
    pd.read_json(f"data/{fname}.json", 
                 orient='records', 
                 lines=True
                ).to_sql(
                        fname,
                        conn,
                        index=False,
                        if_exists='replace')


def clean_offers():
    """Splits out offers from the transcript dataset with transcripts excluded."""
    offers = to_df(
        """select person,
               event,
               json_extract(value, '$.offer_id') offer_id,
               json_extract(value, '$.reward')   reward,
               time
        from (
                 select person, event, replace(value, 'offer id', 'offer_id') value, time
                 from transcript
                 where event like '%offer%'
             );""")
    return (offers.join(pd.get_dummies(offers.event, prefix='event_'), how='outer')
            .drop(columns=['event', 'reward']))

def clean_transcripts():
    """Splits out transcript data from the transcript dataset with offers excluded."""
    return to_df("""select person, 
                               -- event, 
                               json_extract(value, '$.amount') amount, time
                            from transcript
                            where event not like '%offer%';""")


def clean_profile():
    profile = to_df('select * from profile')
    profile.age = profile.age.astype(int)
    
    #CONVERT GENDER TO CATEGORY
#     profile = pd.concat([profile, pd.get_dummies(profile.gender, prefix='gender_')]).drop(columns=['gender'])
#     profile.gender = profile.gender.astype('category')
	profile.gender = profile.gender.map({'M': 1, 'F': 2, 'O': 3, None:3})
    
    #CLEAN DATE FIELD
    conv_date = lambda a: datetime.strptime(str(int(a)), '%Y%m%d').strftime('%m/%d/%Y') if not np.isnan(a) else np.NaN
    profile.became_member_on = profile.became_member_on.apply(conv_date).astype('datetime64')

    #BINNING AGE BRACKETS
    profile = profile[profile.age != 118]
    age_map = {i*10:grp for grp, i in enumerate(range(0,15))}
    profile['age_by_decade'] = profile.age.apply(lambda x: age_map[int(str(x)[:-1] + '0')])
    profile.drop(columns=['age'])
    
    profile.income = profile.income.astype(int)
    return profile

def clean_portfolio():
    """
    Engineers multiple features from the original channel and offer_type variables.
    
    Example adapted from:
    https://stackoverflow.com/questions/29034928/pandas-convert-a-column-of-list-to-dummies
    """
    portfolio = to_df('select * from portfolio')
    portfolio.channels = portfolio.channels.apply(json.loads)
    portfolio.channels = portfolio.channels.apply(lambda x: sorted(x))
    _ = pd.get_dummies(portfolio.channels.apply(pd.Series).stack()).sum(level=0)
    portfolio[list(_)] = _
    portfolio = portfolio.drop(columns=['channels'])
    portfolio = portfolio.join(pd.get_dummies(portfolio.offer_type, prefix='offer_type_'),
                           how='outer').drop(columns=['offer_type'])
    return portfolio

def merge_datasets(offers, transactions, portfolio, profile):
    """
    Merges all the cleaned datasets together to form one joined
    set of features.
    """
    offers_cols = ['person', 'offer_id', 'time', "event__offer completed", 
                   "event__offer received", "event__offer viewed",]
    profile_cols = ['id','gender','age','became_member_on','income',]
    portfolio_cols = ['id','reward','difficulty','duration','email','mobile','social',
    'web','offer_type__bogo','offer_type__discount','offer_type__informational',]

    return (offers[offers_cols].rename(columns={'person':'offer_person_id'})
         .merge(profile.rename(columns={'id':'profile_person_id'}), 
                left_on='offer_person_id',
                right_on='profile_person_id',
                how='inner')
         .merge(portfolio[portfolio_cols].rename(columns={'id':'portfolio_id'}),
                left_on='offer_id',
                right_on='portfolio_id', 
                how='inner')
         .merge(transactions.groupby(by='person')['amount'].mean().reset_index(),
                left_on='profile_person_id',
                right_on='person',
                how='inner'
         )
    )

def clean_merged_dataset(joined):
    """
    Cleanes the resultant merged dataset joined via the 'merge_datasets' function.
    
    NOTE: This function is separate from the previous functions because exploration 
    and visualization is preformed on the merged dataset prior to cleaning
    """
    joined['days_as_customer'] = joined.became_member_on.apply(lambda x: pd.Timestamp.now() - pd.to_datetime(x)).dt.days
    joined['became_member_dayofweek'] = joined.became_member_on.dt.dayofweek
    joined['became_member_month'] = joined.became_member_on.dt.month
    joined['became_member_year'] =  joined.became_member_on.dt.year
    joined = joined.drop(columns=['became_member_on', 'time', 'reward', 'age'])
    return joined

def scale_features(data):
    scaler = MinMaxScaler() # default=(0, 1)
    numerical = list(data.select_dtypes('float'))
#     keys_df = data[['offer_person_id', 'offer_id','profile_person_id', 'portfolio_id']]
#     features = data.drop(columns=[keys_df.columns])
    features = data.drop(columns=['offer_person_id', 'offer_id','profile_person_id', 'portfolio_id'])
    features_log_minmax_transform = pd.DataFrame(data = features)
    features_log_minmax_transform[numerical] = scaler.fit_transform(data[numerical])
    return features_log_minmax_transform


def clean_and_prep_features(features_log_minmax_transform, print_vif=True):
    """This takes the outputted scaled features from the 'scale_features' function,
    drops NaNs and unnecessary columns, and then separates X and Y vars.
    
    NOTE: The dropped variables were determined by running the VIF and retroactively updating this function
    However, I commented this out since RandomForest is the chosen model 
    and as such is robust to multicolinearity."""
    features_log_minmax_transform = features_log_minmax_transform.dropna()
    y=features_log_minmax_transform['event__offer completed']
    features = features_log_minmax_transform.drop(columns=['event__offer completed','person'
#                                                            'event__offer viewed'
    #                                                         'transaction_amount'])
                                                          ])
#     features.drop(columns=['days_as_customer','became_member_month','became_member_year','gender'])
    return features, y
        
def VIF(df):
    """This calculates the Variable Inflation Factor to help eliminate multicolinearity from
    the data.
    
    NOTE: This is useful for exploring and evaluating models, but isn't necessary for
    the eventual chosen model - RandomForest."""
    def calc_vif(X):
        vif = pd.DataFrame()
        vif["variables"] = X.columns
        vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        return(vif)

    vif = calc_vif(df.dropna())
    vif = vif[vif.VIF > 10]
    vif.VIF = vif.VIF.replace([np.inf, -np.inf], np.nan)
    vif = vif.dropna()
    # if print_vif:
    display(vif)
    print(f'Variables to drop with high VIF values: {vif.variables.tolist()}')


def split_and_resample(features, y):
    """
    Adapted from:
    https://towardsdatascience.com/methods-for-dealing-with-imbalanced-data-5b761be45a18
    
    This resamples the dataset to solve the class-imbalance issue at hand.
    
    SIDE NOTE: THe 'class imbalance problem' of
    classification models biases classification towards the majority class
    and results in misleading accuracy scores. 
    I.e. A model can always predict the majority class value and still be
    technically correct the majority of the time but fail to serve the 
    purpose of the model.
    """
    y_var_name = 'event__offer completed'
    # concatenate our training data back together
    X_train, X_test, y_train, y_test = train_test_split(features, 
                                                    y, 
                                                    test_size = 0.2, 
                                                    random_state = 0)
    X = pd.concat([X_train, y_train], axis=1)
    cols = list(X_train); cols.append(y_var_name)
    X.columns = cols
    X[y_var_name] = X[y_var_name].astype(int)
    # separate minority and majority classes
    majority = X[X[y_var_name] == 0.0]
    minority = X[X[y_var_name]== 1]
    # upsample minority
    upsampled = resample(minority,
                              replace=True, # sample with replacement
                              n_samples=len(majority), # match number in majority class
                              random_state=27) # reproducible results

    # combine majority and upsampled minority
    upsampled = pd.concat([majority, upsampled])
    display(upsampled[y_var_name].value_counts())
    
    y_train = upsampled[y_var_name]
    X_train = upsampled.drop(y_var_name, axis=1)
    return X_train, X_test, y_train, y_test

def grid_search(model, params, X, y, X_train, y_train, X_test, y_test):
    """Adapted from my Introduction to ML project"""
    scorer = make_scorer(f1_score)
    g_search = GridSearchCV(model, params,
                              cv=StratifiedKFold(n_splits=5, 
                                                 shuffle = True, 
                                                 random_state = 1001).split(
                                  X, y.values.ravel()), verbose=3, n_jobs=6)
    g_fit = g_search.fit(X, y.values)
    best_model = g_fit.best_estimator_
    predictions = (model.fit(X_train, y_train)).predict(X_test)
    best_predictions = best_model.predict(X_test)
    print("Unoptimized model\n------")
    print("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions)))
    print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta = 0.5)))
    print("\nOptimized Model\n------")
    print("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
    print("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5)))   
    
    return {
        'best_clf': best_model,
        'unoptimized_accuracy_score': accuracy_score(y_test, predictions),
        'unoptimized_beta_score': fbeta_score(y_test, predictions, beta = 0.5),
        'optimized_accuracy_score': accuracy_score(y_test, best_predictions),
        'optimized_beta_score': fbeta_score(y_test, best_predictions, beta = 0.5),
    }

def train_and_plot(model, X_train, y_train, X_test, y_test, show_plots=True):
    """
    This trains and plots resulting metrics for a given model and it's split datasets.
    
    DISCUSSION:
    F1 scores, precision, recall, and ROC AUC are used as metrics in addition 
    to accuracy scores particularly because accuracy scores are misleading
    for classification models with imbalanced classes.
    
    ADAPTED FROM:
    https://machinelearningmastery.com/
    roc-curves-and-precision-recall-curves-for-classification-in-python/
    """
    start = time()
    model.fit(X_train[X_train.notnull()], y_train[y_train.notnull()].values.ravel())
    model.predict(X_test)
    predictions_train = model.predict(X_train) 
    predictions_test = model.predict(X_test)

    ns_probs = [0 for _ in range(len(y_test))]
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)
    probs = probs[:, 1]
    ns_auc = roc_auc_score(y_test, ns_probs)
    auc = roc_auc_score(y_test, probs)
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    fpr, tpr, _ = roc_curve(y_test, probs)
    
    end = time() - start
    
    ptrain, rtrain, ftrain, _ = precision_recall_fscore_support(y_train, predictions_train,
                                                 average='macro')
    ptest, rtest, ftest, _ = precision_recall_fscore_support(y_test, predictions_test,
                                                 average='macro')
    creport = pd.DataFrame(classification_report(y_test, predictions_test, output_dict=True)).transpose()
    model_name = model.__class__.__name__
    results_df = pd.DataFrame([
    (model_name, 'model_score', model.score(X_test, y_test), end),
    (model_name, 'acc_train', accuracy_score(y_train, predictions_train), end),
    (model_name, 'acc_test', accuracy_score(y_test, predictions_test), end),
    (model_name, 'f_train', fbeta_score(y_train, predictions_train, beta=0.5), end),
    (model_name, 'f_test', fbeta_score(y_test, predictions_test, beta=0.5), end),
    (model_name, 'precision_train', ptrain, end),
    (model_name, 'precision_test', ptest, end),
    (model_name, 'recall_train', rtrain, end),
    (model_name, 'recall_test', rtest, end),
    (model_name, 'f1_train', ftrain, end),
    (model_name, 'f1_test', ftest, end),
    (model_name, 'No Skill: ROC AUC', ns_auc, end),
    (model_name, 'ROC AUC', auc, end),
    ], columns=['Model','Metric', 'Score', 'Training_time'])
    
    if show_plots:
        display(creport)
        plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
        plt.plot(fpr, tpr, marker='.', label=model_name)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.show()
    return results_df