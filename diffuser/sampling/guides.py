import torch
import torch.nn as nn
import pdb
from abc import abstractclassmethod
import numpy as np

class Guide(nn.Module):
	"""
	Abstract class for guide
		gradients: return the gradients of the guide, input: x
		metrics: return the metrics of the guide, input: x
			e.g. {
				"distance": 1.0,
				"velocity": 1.0,
			}
	"""
	def gradients(self, x, **kwargs):
		raise NotImplementedError

	def metrics(self, x, **kwargs):
		raise NotImplementedError

class ValueGuide(Guide):

	def __init__(self, model):
		super().__init__()
		self.model = model

	def forward(self, x, cond, t):
		output = self.model(x, cond, t)
		return output.squeeze(dim=-1)

	def gradients(self, x, *args):
		x.requires_grad_()
		y = self(x, *args)
		grad = torch.autograd.grad([y.sum()], [x])[0]
		x.detach()
		return y, grad


class NoTrainGuide(Guide):
	
	@abstractclassmethod
	def forward(self, x, cond, t):
		pass

	def gradients(self, x, *args):
		x.requires_grad_()
		y = self(x, *args)
		grad = torch.autograd.grad([y.sum()], [x])[0]
		x.detach()
		return y, grad

class NoTrainGuideDistance(NoTrainGuide):
	"""
	make the total distance of the trajectory become shorter or longer
	"""
	SHOTER = None

	def forward(self, x, cond, t):
		"""
		cal total distance
		x: (batch_size, trace_length, 6)
			the last dim is [x, y, vx, vy, act_x, act_y]
			we only use x, y to calculate distance
		"""
		total_distance = self.cal_distance(x)

		if self.SHOTER is None: raise NotImplementedError("SHOTER is not defined")
		if self.SHOTER: return -total_distance
		else: return total_distance
	
	def cal_distance(self, x):
		"""
		cal total distance
		x: (batch_size, trace_length, 6)
			the last dim is [x, y, vx, vy, act_x, act_y]
			we only use x, y to calculate distance
		"""
		# Extract x, y coordinates
		coord = x[:, :, :2]

		# Compute differences between successive coordinates along the trace
		sqdist = ((coord[:, 1:, :] - coord[:, :-1, :])**2).sum(dim=-1)

		# Compute total distance
		total_distance = sqdist.sum(dim=-1)

		return total_distance
	
	def metrics(self, x, **kwargs):
		# if x is numpy array, convert it to torch tensor
		if isinstance(x, np.ndarray): x = torch.from_numpy(x)
		with torch.no_grad():
			return {
				"distance": self.cal_distance(x),
			}


class NoTrainGuideShorter(NoTrainGuideDistance):
	SHOTER = True

class NoTrainGuideLonger(NoTrainGuideDistance):
	SHOTER = False

class NoTrainGuideRepeat(NoTrainGuide):
	"""
	Make two middle points become the same, so that it will go back
	to the same point.
	Default: the point at 1/3 and 2/3
	"""

	def forward(self, x, cond, t):
		"""
		Calculate the squared Euclidean distance between the points at 1/3 and 2/3 of the trajectory.
		x: (batch_size, trace_length, 6)
			the last dim is [x, y, vx, vy, act_x, act_y]
			we only use x, y to calculate distance
		"""
		# Extract x, y coordinates
		coord = x[:, :, :2]  # shape: (batch_size, trace_length, 2)

		# Compute indices for 1/3 and 2/3 of the trajectory
		idx1 = coord.shape[1] // 3
		idx2 = 2 * idx1
		
		# Get the points at 1/3 and 2/3 of the trajectory
		point1 = coord[:, idx1, :]  # shape: (batch_size, 2)
		point2 = coord[:, idx2, :]  # shape: (batch_size, 2)

		# Compute the squared Euclidean distance between point1 and point2
		sqdist = ((point1 - point2)**2).sum(dim=-1)  # shape: (batch_size,)

		return - sqdist