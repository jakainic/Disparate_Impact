import pandas as pd 
import numpy as np
import csv

#drop protected features that we aren't repairing
#drop columns that don't have a natural ordering
#encode qualitative feats into their corresponding numerical values
#normalize

#protected feats is dict with keys: (target_header, target_maj_class, target_min_classes, other)
#outcomes is dict with keys: (header, preferred, other)
def process_data(data, protected_feats, cat_orderings, unordered_cols, outcomes):

	data.drop(protected_feats["other"],axis=1,inplace=True)
	data.drop(unordered_cols, axis=1, inplace=True)

	data[outcomes["header"]].replace(outcomes["preferred"],1,inplace=True)
	data[outcomes["header"]].replace(outcomes["other"],-1,inplace = True)


	data[protected_feats["target_header"]].replace(protected_feats["target_maj_class"], 1, inplace=True)
	data[protected_feats["target_header"]].replace(protected_feats["target_min_classes"], -1, inplace=True)


	for col in cat_orderings.keys():
		categories = [el[0] for el in cat_orderings[col]]
		encoding = [el[1] for el in cat_orderings[col]]
		data[col].replace(categories, encoding, inplace=True)


	data = data.astype(float)

	for col in data.columns:
		data[col]= 2*(data[col]-data[col].min())/(data[col].max()-data[col].min()) -1

	return data
