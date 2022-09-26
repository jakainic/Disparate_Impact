import pandas as pd
import numpy as np

#a bit messy, but essentially this is all the information the data processing function needs to get rid of some columns
#and encode others as numbers
#also, reading the data

path = 'german_credit.csv'
protected_feats = {"target_header": "age", "target_maj_class": "old", "target_min_classes": "young","other": ["personal_status"]}
cat_orderings = {'checking_status': [('no checking',0), ('<0',1), ('0<=X<200',2), ('>=200',3)], 
'credit_history': [('critical/other existing credit',0), ('existing paid',2), ('delayed previously',1), ('no credits/all paid',4), ('all paid',3)],
'savings_status': [('no known savings',0), ('<100',1), ('100<=X<500',2),('500<=X<1000',3),('>=1000',4)],
'employment': [('unemployed',0), ('<1',1), ('1<=X<4',2),('4<=X<7',3),('>=7',4)],
'other_parties': [('none',0),('co applicant',1),('guarantor',2)],
'property_magnitude': [('no known property',0), ('car',1), ('life insurance',2),('real estate',3)],
'other_payment_plans': [('none',0), ('stores',1), ('bank',2)],
'housing': [('for free',0), ('own',1),('rent',2)],
'job': [('unemp/unskilled non res',0),('unskilled resident',1),('skilled',2),('high qualif/self emp/mgmt',3)],
'own_telephone': [('none',0),('yes',1)],
'foreign_worker': [('yes',1),('no',0)]}
outcomes = {"header":'class', "preferred": 'good',"other":'bad'}
unordered_cols = ['purpose']

protected_feat = protected_feats["target_header"]
outcome = outcomes["header"]
###

data = pd.read_csv(path)

#protected feat to binary
data[protected_feat]=np.where(data[protected_feat]>25, "old", "young")

inputs = [col for col in data.columns if col!=protected_feat and col!= outcomes["header"] and col not in unordered_cols+protected_feats["other"]]
feats = inputs+[protected_feat]
