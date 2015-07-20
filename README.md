# ud120-finalproject
Using Machine Learning to Identify Enron Corporate Fraud using enron dataset.

Working with Enron Datasets.ipynb
Features:
What is the person of interest:
-indicted 
-settled without admitting guilt
-testified in exchange for immunity

The Data Set:
Large number of e-mails

Types of Data:
numerical = numerical values (numbers)

categorical = limited number of discrete values (category)

time series = temoral value (date, timestamp)

text = words

The form of the dataset:

enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

features_dict contains features associated with a person.


Enron Data-
The interesting and hard part of the dataset is that the distribution of the non-POI's to POI's is very skewed,
given that from the 146 there are only 11 people or data points labeled as POI's or guilty of fraud. We are 
interested in labeling every person in the dataset into either a POI or a non-POI (POI stands for Person Of Interest).
More than that, if we can assign a probability to each person to see what is the chance she is POI, it would 
be a much more reasonable model given that there is always some uncertainty.



