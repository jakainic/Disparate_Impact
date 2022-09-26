from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import balanced_accuracy_score
from disparate_impact import compute_di
import pandas as pd

def predict(model_type, feats, outcomes, protected_class_name):
	if model_type == 'SVM':
		return predict_SVM(feats, outcomes, protected_class_name)
	elif model_type == 'LR':
		return predict_LR(feats, outcomes, protected_class_name)
	elif model_type == 'GNB':
		return predict_GNB(feats,outcomes, protected_class_name)
	else:
		print("Error: model type not recognized")
		return -1,-1

def predict_GNB(feats, outcomes, protected_class_name):
	X_train, X_test, y_train, y_test = train_test_split(feats, outcomes,test_size = 0.25, random_state = 1)
	X_train.drop(protected_class_name, axis=1, inplace=True)

	model = GaussianNB(priors = [0.5,0.5])
	model.fit(X_train,y_train)
	y_pred = model.predict(X_test.drop(protected_class_name, axis=1, inplace=False))

	utility = balanced_accuracy_score(y_test,y_pred)
	fairness = compute_di(X_test[protected_class_name].to_numpy(), y_pred)

	return utility, fairness

def predict_SVM(feats, outcomes, protected_class_name):
	X_train, X_test, y_train, y_test = train_test_split(feats, outcomes,test_size = 0.25, random_state = 1)
	X_train.drop(protected_class_name, axis=1, inplace=True)
	C = best_param_SVM(X_train, y_train)

	model = LinearSVC(penalty = 'l2', loss = 'squared_hinge',C=C, class_weight = 'balanced', dual = False)
	model.fit(X_train,y_train)
	y_pred = model.predict(X_test.drop(protected_class_name, axis=1, inplace=False))

	utility = balanced_accuracy_score(y_test,y_pred)
	fairness = compute_di(X_test[protected_class_name].to_numpy(), y_pred)

	return utility, fairness

def best_param_SVM(train_in, train_out):
	best_C=1
	best_score = -1
	
	for exp in [k/2 for k in range(-6,7)]:
		model = LinearSVC(penalty = 'l2', loss = 'squared_hinge',C=10**exp, class_weight = 'balanced',dual = False)
		score = cross_validate(model, train_in, train_out, cv=3, scoring='balanced_accuracy')["test_score"].mean()

		if best_score < 0 or score>best_score:
			best_C = 10**exp
			best_score = score

	return best_C

def predict_LR(feats, outcomes, protected_class_name):
	X_train, X_test, y_train, y_test = train_test_split(feats, outcomes,test_size = 0.25, random_state = 1)
	X_train.drop(protected_class_name, axis=1, inplace=True)
	C = best_param_LR(X_train, y_train)

	model = LogisticRegression(penalty = 'l2', C = C, class_weight = 'balanced', solver = 'liblinear',dual = False)
	model.fit(X_train,y_train)
	y_pred = model.predict(X_test.drop(protected_class_name, axis=1, inplace=False))

	utility = balanced_accuracy_score(y_test,y_pred)
	fairness = compute_di(X_test[protected_class_name].to_numpy(), y_pred)

	return utility, fairness

def best_param_LR(train_in, train_out):
	best_C=1
	best_score = -1
	
	for exp in [k/2 for k in range(-6,7)]:
		model = LogisticRegression(penalty = 'l2', C=10**exp, class_weight = 'balanced',solver = 'liblinear',dual = False)
		score = cross_validate(model, train_in, train_out, cv=3, scoring='balanced_accuracy')["test_score"].mean()

		if best_score < 0 or score>best_score:
			best_C = 10**exp
			best_score = score

	return best_C