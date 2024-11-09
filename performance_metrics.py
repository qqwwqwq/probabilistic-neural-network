# Wrote my own implementations of performance metrics just for fun
# Feel free to use the scikit-learn functions

import numpy as np

# Valid only for two class cases

def confusion_matrix(target, predicted, num_class):

	target_nb = target
	predicted_nb = predicted
	
	num_trainset = target_nb.size

	cm = np.zeros((num_class, num_class))
	for i in range(num_trainset):
		cm[target_nb[i], predicted_nb[i]] += 1
	
	return cm

def accuracy(target, predicted):

	temp = target - predicted
	non_zero = np.count_nonzero(temp)
	zero = target.shape[0] - non_zero

	return float(zero) / target.shape[0]

def precision(target, predicted, num_class):
	
	num_trainset = target.shape[0]
	cm = confusion_matrix(target, predicted, num_class)

	p = np.zeros((num_class, 1))

	for i in range(num_class):
		p[i] = cm[i, i]/np.sum(cm[:, i])

	return np.mean(p)

def recall(target, predicted, num_class):
	num_trainset = target.shape[0]
	cm = confusion_matrix(target, predicted, num_class)

	r = np.zeros((num_class, 1))

	for i in range(num_class):
		r[i] = cm[i, i]/np.sum(cm[i, :])

	return np.mean(r)
