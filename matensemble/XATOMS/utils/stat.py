import numpy as np

def get_probability(data, target_window):
	from scipy.stats import gaussian_kde
	kde = gaussian_kde(data, bw_method=0.1)
	n_discrete = 1000

	assert target_window[0]<target_window[1], "target_window should consist of the bounds [x_min, x_max]"
	
	window = target_window[1] - target_window[0]
	# while (window > (max(theta) - min(theta)+2)/n_discrete):xx
	# 	n_discrete*=100
	
	x_values = np.linspace(min(data)-1, max(data)+1, n_discrete)

	# Evaluate the PDF for the range of x values
	pdf_values = kde(x_values)

	# Compute the CDF by integrating the PDF (numerical integration)
	cdf_values = np.cumsum(pdf_values) * (x_values[1] - x_values[0])  # Multiply by dx to approximate the integral

	# Normalize the CDF to make sure it ranges from 0 to 1
	cdf_values /= cdf_values[-1]

	cdf_window = cdf_values[np.argwhere((x_values>=target_window[0]) & (x_values<=target_window[1]))]

	try:
		# compute probaility
		prob = cdf_window[-1] - cdf_window[0]
	except:
		# for no samples in cdf_window
		prob = np.array([0])
	return prob