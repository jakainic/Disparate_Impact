def get_group_data(repair_col, protected_col, gps):
	num_rows = len(repair_col)
	gp_data = {gp: [] for gp in gps}


	for row in range(num_rows):
		gp = protected_col[row]
		gp_data[gp].append((repair_col[row],row))

	for gp in gps:
		gp_data[gp].sort(key = lambda entry: entry[0])

	return gp_data

def make_quantiles(gp_data):
	gps = gp_data.keys()
	unique_gp_vals = {gp: set([val for (val, index) in gp_data[gp]]) for gp in gps}
	num_quantiles = min([len(unique_gp_vals[gp]) for gp in gps])

	grouped_quantile_indices = {q: {gp:[] for gp in gps} for q in range(num_quantiles)}
	grouped_quantile_values = {q: {gp: [] for gp in gps} for q in range(num_quantiles)}
	num_in_gp = {gp: len(gp_data[gp]) for gp in gps}

	for q in range(num_quantiles):
		last_quantile = False if q<num_quantiles-1 else True

		for gp in gps:
			num_to_put = int(round(num_in_gp[gp]/num_quantiles))
			entries_to_put = gp_data[gp][q*num_to_put:(q+1)*num_to_put] if not last_quantile else gp_data[gp][q*num_to_put:-1]
		
			grouped_quantile_indices[q][gp] = [index for (value, index) in entries_to_put]
			grouped_quantile_values[q][gp] = [value for (value, index) in entries_to_put]

	return grouped_quantile_indices, grouped_quantile_values