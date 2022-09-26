import pandas as pd
import numpy as np
import math
from statistics import median_low
from repair_help import *

#FOR NOW: everything is with a single, binary protected variable, so don't have to worry (e.g.) about stratifying with respect to several protected classes

#geometric repair with binning for target distribution
def binned_repair_all_geo(data,protected_feat,degree):
	new_data = pd.DataFrame()

	for col in data.columns:
		if col!= protected_feat:
			new_data[col] = binned_repair_col_geo(data,col, protected_feat,degree)
		if col == protected_feat:
			new_data[col] = data[col]

	return new_data

#Get position from all valid values for an attribute from what is in the quantile
#and median for quantile selected from representative medians
def binned_repair_all_comb1(data, protected_feat, degree):
	new_data = pd.DataFrame()

	for col in data.columns:
		if col!= protected_feat:
			new_data[col] = binned_repair_col_comb1(data,col, protected_feat,degree)
		if col == protected_feat:
			new_data[col] = data[col]

	return new_data

#Get position from all valid values for an attribute (but still look at median from representative medians in quantile)
def binned_repair_all_comb2(data, protected_feat, degree):
	new_data = pd.DataFrame()

	for col in data.columns:
		if col!= protected_feat:
			new_data[col] = binned_repair_col_comb2(data,col, protected_feat,degree)
		if col == protected_feat:
			new_data[col] = data[col]

	return new_data

#repair towards distribution that is a Wasserstein-1 barycenter
def OT_repair_all(data, protected_feat, degree, type_of_repair):
	new_data = pd.DataFrame()

	for col in data.columns:
		if col!= protected_feat:
			new_data[col] = OT_repair_col(data,col, protected_feat,degree,type_of_repair)
		if col == protected_feat:
			new_data[col] = data[col]

	return new_data

###

def binned_repair_col_geo(data,feat_to_repair,protected_feat,degree):
	gps = [-1,1]
	repair_col = data[feat_to_repair].to_numpy()
	protected_col = data[protected_feat].to_numpy()

	gp_data = get_group_data(repair_col, protected_col, gps)
	grouped_quantile_indices, grouped_quantile_values = make_quantiles(gp_data)

	for quantile in grouped_quantile_indices.keys():
		gp_medians = {gp: median_low(grouped_quantile_values[quantile][gp]) for gp in gps}
		quantile_median = median_low(gp_medians.values())

		for gp in gps:
			for row in grouped_quantile_indices[quantile][gp]:
				curr_val = repair_col[row]
				new_val = (1-degree)*curr_val + degree*quantile_median

				repair_col[row] = new_val

	return repair_col

def binned_repair_col_comb1(data,feat_to_repair,protected_feat,degree):
	gps = [-1,1]
	repair_col = data[feat_to_repair].to_numpy()
	protected_col = data[protected_feat].to_numpy()

	gp_data = get_group_data(repair_col, protected_col, gps)
	grouped_quantile_indices, grouped_quantile_values = make_quantiles(gp_data)

	for quantile in grouped_quantile_indices.keys():
		gp_medians = {gp: median_low(grouped_quantile_values[quantile][gp]) for gp in gps}
		quantile_median = median_low(gp_medians.values())
		
		unique_quantile_vals = sorted(list(set([value for gp in gps for value in grouped_quantile_values[quantile][gp]])))

		for gp in gps:
			for row in grouped_quantile_indices[quantile][gp]:
				curr_val = repair_col[row]
				curr_pos = unique_quantile_vals.index(curr_val)
				quantile_median_pos = unique_quantile_vals.index(quantile_median)
				
				new_pos = curr_pos + int(round(degree*(quantile_median_pos-curr_pos)))
				new_val = unique_quantile_vals[new_pos]
				
				repair_col[row] = new_val

	return repair_col

def binned_repair_col_comb2(data,feat_to_repair,protected_feat,degree):
	gps = [-1,1]
	repair_col = data[feat_to_repair].to_numpy()
	protected_col = data[protected_feat].to_numpy()
	all_unique_vals = sorted(list(set(repair_col)))

	gp_data = get_group_data(repair_col, protected_col, gps)
	grouped_quantile_indices, grouped_quantile_values = make_quantiles(gp_data)

	for quantile in grouped_quantile_indices.keys():
		gp_medians = {gp: median_low(grouped_quantile_values[quantile][gp]) for gp in gps}
		quantile_median = median_low(gp_medians.values())
		
		for gp in gps:
			for row in grouped_quantile_indices[quantile][gp]:
				curr_val = repair_col[row]
				curr_pos = all_unique_vals.index(curr_val)
				quantile_median_pos = all_unique_vals.index(quantile_median)
				
				new_pos = curr_pos + int(round(degree*(quantile_median_pos-curr_pos)))
				new_val = all_unique_vals[new_pos]

				repair_col[row] = new_val

	return repair_col

#def OT_repair_col(data, feat_to_repair, protected_feat, degree, type_of_repair):
