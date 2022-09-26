import pandas as pd
import numpy as np

#assumes both inputs are arrays, not pandas dataframes
def gp_selection_rate(protected_data, outcomes, gp):
	num_in_gp = np.count_nonzero(protected_data == gp)
	num_in_gp_selected = len([i for i, el in enumerate(protected_data) if el==gp and outcomes[i]==1])

	return num_in_gp_selected/num_in_gp


def compute_di(protected_data, outcomes):
	return gp_selection_rate(protected_data,outcomes,-1)/gp_selection_rate(protected_data,outcomes,1)