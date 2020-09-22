import numpy as np

mean, std = np.random.rand(), np.random.rand()

for i in range(1, 114):

	# Create adjacency matrices

	t0 = np.abs(np.random.normal(mean, std, (35,35))) % 1.0
	mean_s = mean + np.random.rand() % 0.1
	std_s = std + np.random.rand() % 0.1
	t1 = np.abs(np.random.normal(mean_s, std_s, (35,35))) % 1.0
	mean_s = mean + np.random.rand() % 0.1
	std_s = std + np.random.rand() % 0.1
	t2 = np.abs(np.random.normal(mean_s, std_s, (35,35))) % 1.0

	# Make them symmetric

	t0 = (t0 + t0.T)/2
	t1 = (t1 + t1.T)/2
	t2 = (t2 + t2.T)/2

	# Clean the diagonals
	t0[np.diag_indices_from(t0)] = 0
	t1[np.diag_indices_from(t1)] = 0
	t2[np.diag_indices_from(t2)] = 0

	# Save them
	s = "cortical.lh.ShapeConnectivityTensor_OAS2_"
	if i < 10:
		s += "0"
	s += "00" + str(i) + "_MR1"

	t0_s = s + "_t0.txt"
	t1_s = s + "_t1.txt"
	t2_s = s + "_t2.txt"

	np.savetxt(t0_s, t0)
	np.savetxt(t1_s, t1)
	np.savetxt(t2_s, t2)
