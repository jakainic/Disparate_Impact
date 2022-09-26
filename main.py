from prediction import *
from data_processing import *
from repair import *
from disparate_impact import *
from german_info import *
import matplotlib.pyplot as plt
import numpy as np


data = process_data(data, protected_feats, cat_orderings, unordered_cols, outcomes)

type_of_repair = 'geo'
degrees = [k/10 for k in range(11)]

for predictor in ['SVM','LR','GNB']:
	utilities = []
	fairnesses = []

	for degree in degrees:

		if type_of_repair == 'geo':
			repaired_data = binned_repair_all_geo(data[feats], protected_feat, degree)
		elif type_of_repair == 'comb1':
			repaired_data = binned_repair_all_comb1(data[feats], protected_feat, degree)
		elif type_of_repair == 'comb2':
			repaired_data = binned_repair_all_comb2(data[feats], protected_feat, degree)
		#elif type_of_repair == 'OT':
		#	repaired_data = 
		else:
			print('Error: type of repair not recognized')
			repaired_data = data[feats]

		utility, fairness = predict(predictor, repaired_data[feats], data[outcome], protected_feat)
		
		utilities.append(utility)
		fairnesses.append(fairness)
	
	plt.scatter(fairnesses, utilities)
	for i, degree in enumerate(degrees):
		plt.annotate(degree, (fairnesses[i], utilities[i]))
	plt.xlabel("Fairness")
	plt.ylabel("Utility")
	plt.title(type_of_repair+ " repair; " + predictor+ " predictor")

	plt.show()