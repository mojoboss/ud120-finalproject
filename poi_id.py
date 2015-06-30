#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

'''
#Full feature list
features_list = ['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments',
                 'loan_advances', 'bonus', 'restricted_stock_deferred',
                 'deferred_income', 'total_stock_value', 'expenses', 'from_poi_to_this_person',
                 'exercised_stock_options', 'from_messages', 'other', 'from_this_person_to_poi',
                 'long_term_incentive', 'shared_receipt_with_poi', 'restricted_stock',
                 'director_fees', 'to_fraction', 'from_fraction'] # You will need to use more features

'''
'''
#Best features
features_list = ['poi',  'deferral_payments','deferred_income', 'expenses',
                 restricted_stock', 'to_fraction', 'from_fraction'] 
'''

features_list = ['poi',  'deferral_payments', 'deferred_income',  'expenses',
                  'restricted_stock', 'to_fraction', 'from_fraction']

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )
#Changing 'NaN' values to 0

for key in data_dict:
    for keyc in data_dict[key]:
        if data_dict[key][keyc]=='NaN':
            data_dict[key][keyc] = 0

### Task 2: Remove outliers
data_dict.pop('TOTAL') # As 'TOTAL  is clearly an outlier'

### Task 3: Create new feature(s)

for key in data_dict:
    if data_dict[key]['to_messages']==0:
        data_dict[key]['to_fraction'] = 0
    else:
        data_dict[key]['to_fraction'] = float(data_dict[key]['from_poi_to_this_person'])/(data_dict[key]['to_messages'])
        
    if data_dict[key]['from_messages']==0:
        data_dict[key]['from_fraction'] = 0
    else:
        data_dict[key]['from_fraction'] = float(data_dict[key]['from_this_person_to_poi'])/(data_dict[key]['from_messages'])
    
### Store to my_dataset for easy export below.
my_dataset = data_dict
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

from sklearn import tree
clf = tree.DecisionTreeClassifier(min_samples_split=45, criterion = 'entropy')    # Provided to give you a starting point. Try a varity of classifiers.
 
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

test_classifier(clf, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)