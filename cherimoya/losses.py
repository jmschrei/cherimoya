# losses.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>


"""
This module contains the losses used by Cherimoya models for training.
"""

import torch

from bpnetlite.losses import MNLLLoss
from bpnetlite.losses import log1pMSELoss

def _mixture_loss(y, y_hat_logits, y_hat_logcounts, labels=None):
	"""A function that takes in predictions and truth and returns the loss.
	
	This function takes in the observed integer read counts, the predicted logits,
	and the predicted logcounts, and returns the total loss. Importantly, this
	calculates a single multinomial over all strands in the tracks and a single
	count loss across all tracks.
	
	The logits do not have to be normalized.
	
	
	Parameters
	----------
	y: torch.Tensor, shape=(n,
	"""
	
	y_hat_logits = y_hat_logits.reshape(y_hat_logits.shape[0], -1)
	y_hat_logits = torch.nn.functional.log_softmax(y_hat_logits, dim=-1)
	
	y = y.reshape(y.shape[0], -1)
	y_ = y.sum(dim=-1).reshape(y.shape[0], 1)

	# Calculate the profile and count losses
	if labels is not None:
		profile_loss = MNLLLoss(y_hat_logits[labels == 1], y[labels == 1]).mean()
	else:
		profile_loss = MNLLLoss(y_hat_logits, y).mean()

	count_loss = log1pMSELoss(y_hat_logcounts, y_).mean()	
	return profile_loss, count_loss
