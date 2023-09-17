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
	def __init__(self, **kwargs):
		super().__init__()
		self.kwargs = kwargs

	def gradients(self, x, **kwargs):
		raise NotImplementedError

	def metrics(self, x, **kwargs):
		raise NotImplementedError

class MultiGuide(Guide):
	def __init__(self, guides):
		super().__init__()
		self.guides = guides
	
	def gradients(self, x, *args):
		"""
		return the gradients of the guide, input: x
		"""
		grads = []
		for guide in self.guides:
			grads.append(guide.gradients(x, *args))
		return grads

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
		x.detach_()
		return y, grad

class NoTrainGuide(Guide):
	
	@abstractclassmethod
	def forward(self, x, cond, t):
		pass

	def gradients(self, x, *args):
		x.requires_grad_()
		y = self(x, *args)
		grad = torch.autograd.grad([y.sum()], [x])[0]
		x.detach_()
		return y, grad

## Distance

class NoTrainGuideDistance(NoTrainGuide):
	"""
	make the total distance of the trajectory become shorter or longer
	"""
	SHORTER = None
	HALF = False

	def forward(self, x, cond, t):
		"""
		cal total distance
		x: (batch_size, trace_length, 6)
			the last dim is [x, y, vx, vy, act_x, act_y]
			we only use x, y to calculate distance
		"""
		total_distance = self.cal_distance(x)

		if self.SHORTER is None: raise NotImplementedError("SHOTER is not defined")
		if self.SHORTER: return -total_distance
		else: return total_distance
	
	def cal_distance(self, x):
		"""
		cal total distance
		x: (batch_size, trace_length, 6)
			the last dim is [act_x, act_y, x, y, vx, vy]
			we only use x, y to calculate distance
		"""
		# Extract x, y coordinates
		coord = x[:, :, 2:4]

		# if HALF, only use the first half of the trajectory
		if self.HALF: coord = coord[:, :coord.shape[1]//2, :]

		# Compute differences between successive coordinates along the trace
		# sqdist = ((coord[:, 1:, :] - coord[:, :-1, :])**2).sum(dim=-1)
		# absolute distance sum
		sqdist = (coord[:, 1:, :] - coord[:, :-1, :]).abs().sum(dim=-1)

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
	SHORTER = True

class NoTrainGuideLonger(NoTrainGuideDistance):
	SHORTER = False

class NoTrainGuideHalfShorter(NoTrainGuideDistance):
	SHORTER = True
	HALF = True

class NoTrainGuideHalfLonger(NoTrainGuideDistance):
	SHORTER = False
	HALF = True

class DummyGuide(NoTrainGuideShorter):
	"""
	always return 0
	"""
	SHORTER = False
	HALF = False

	def forward(self, x, cond, t):
		total_distance = self.cal_distance(x)
		
		if self.SHORTER is None: raise NotImplementedError("SHOTER is not defined")
		if self.SHORTER: return -total_distance * 0
		else: return total_distance *  0.0

## Average coordinate

class NoTrainGuideAvgCoordinate(NoTrainGuide):
	"""
	Base class to make the average coordinate (x or y) lower or higher
	"""
	LOWER = None
	COORDINATE = None

	def forward(self, x, cond, t):
		"""
		Calculate average coordinate (x or y)
		x: (batch_size, trace_length, 6)
			the last dim is [act_x, act_y, x, y, vx, vy]
			we only use x or y to calculate average coordinate
		"""
		avg_coordinate = self.cal_average_coordinate(x)

		if self.LOWER is None: raise NotImplementedError("LOWER is not defined")
		if self.COORDINATE is None: raise NotImplementedError("COORDINATE is not defined")
		if self.LOWER: return - avg_coordinate
		else: return avg_coordinate

	def cal_average_coordinate(self, x):
		"""
		Calculate average coordinate (x or y)
		x: (batch_size, trace_length, 6)
			the last dim is [act_x, act_y, x, y, vx, vy]
			we only use x or y to calculate average coordinate
		"""
		# Extract x or y coordinate
		coord = x[:, :, self.COORDINATE+2]

		# Compute average coordinate
		avg_coordinate = coord.mean(dim=1)

		return avg_coordinate

	def metrics(self, x, **kwargs):
		"""
		Calculate average coordinate (x and y)
		x: (batch_size, trace_length, 6)
			the last dim is [x, y, vx, vy, act_x, act_y]
			we use x and y to calculate average coordinates
		"""
		# if x is numpy array, convert it to torch tensor
		if isinstance(x, np.ndarray): x = torch.from_numpy(x)
		
		avg_x = x[:, :, 0].mean(dim=1)
		avg_y = x[:, :, 1].mean(dim=1)

		return {
			"avg_x": avg_x,
			"avg_y": avg_y
		}

class NoTrainGuideXLower(NoTrainGuideAvgCoordinate):
	LOWER = True
	COORDINATE = 0

class NoTrainGuideXHigher(NoTrainGuideAvgCoordinate):
	LOWER = False
	COORDINATE = 0

class NoTrainGuideYLower(NoTrainGuideAvgCoordinate):
	LOWER = True
	COORDINATE = 1

class NoTrainGuideYHigher(NoTrainGuideAvgCoordinate):
	LOWER = False
	COORDINATE = 1

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
	
	def metrics(self, x, **kwargs):
		# if x is numpy array, convert it to torch tensor
		if isinstance(x, np.ndarray): x = torch.from_numpy(x)
		with torch.no_grad():
			coord = x[:, :, :2]  # shape: (batch_size, trace_length, 2)

			# Compute indices for 1/3 and 2/3 of the trajectory
			idx1 = coord.shape[1] // 3
			idx2 = 2 * idx1
			
			# Get the points at 1/3 and 2/3 of the trajectory
			point1 = coord[:, idx1, :]  # shape: (batch_size, 2)
			point2 = coord[:, idx2, :]  # shape: (batch_size, 2)

			# Compute the squared Euclidean distance between point1 and point2
			distance = ((point1 - point2)**2).sum(dim=-1)  # shape: (batch_size,)
			
			return {
				"distance_repeat": distance
			}

## Mujoco Distance
class NoTrainGuideOffset(NoTrainGuide):
	"""
	make the total distance of the trajectory become shorter or longer
	"""
	LOWER = None
	HALF = False

	def forward(self, x, cond, t):
		"""
		cal total distance
		x: (batch_size, trace_length, 6)
			the last dim is [x, y, vx, vy, act_x, act_y]
			we only use x, y to calculate distance
		"""
		total_distance = self.cal_value(x)

		if self.LOWER is None: raise NotImplementedError("SHOTER is not defined")
		if self.LOWER: return -total_distance
		else: return total_distance
	
	def cal_value(self, x):
		"""
		cal total distance
		x: (batch_size, trace_length, 6)
			the last dim is [act*6, root_x, root_y, root_vx, root_vy, ...]
			we only use x, y to calculate distance
		"""
		# Extract x, y coordinates
		ACT_DIM = 6
		z = x[:, :, ACT_DIM+0] # height
		vx = x[:, :, ACT_DIM+8] # v horizontal
		vz = x[:, :, ACT_DIM+9] # v vertical/height
		total_distance = vx.mean(dim=1)
		return total_distance
	
	def metrics(self, x, **kwargs):
		# if x is numpy array, convert it to torch tensor
		if isinstance(x, np.ndarray): x = torch.from_numpy(x)
		with torch.no_grad():
			return {
				"offset": self.cal_value(x),
			}

class NoTrainGuideCloser(NoTrainGuideOffset):
	LOWER = True

class NoTrainGuideFarther(NoTrainGuideOffset):
	LOWER = False

## Mujoco Height
class SingleValueGuide(NoTrainGuide):
	INDEX = None # z:6, vx:14, vz:15
	LOWER = None
	NAME = None

	def forward(self, x, cond, t):
		"""
		cal total distance
		x: (batch_size, trace_length, 6)
			the last dim is [x, y, vx, vy, act_x, act_y]
			we only use x, y to calculate distance
		"""
		assert self.INDEX is not None, "INDEX is not defined"
		assert self.LOWER is not None, "LOWER is not defined"
		assert self.NAME is not None, "NAME is not defined"

		value = self.cal_value(x)

		if self.LOWER is None: raise NotImplementedError("SHOTER is not defined")
		if self.LOWER: return -value
		else: return value
	
	def cal_value(self, x):
		"""
		cal total distance
		x: (batch_size, trace_length, 6)
			the last dim is [act*6, root_x, root_y, root_vx, root_vy, ...]
			we only use x, y to calculate distance
		"""
		# Extract x, y coordinates
		# ACT_DIM = 6
		# z = x[:, :, ACT_DIM+0] # height
		# vx = x[:, :, ACT_DIM+8] # v horizontal
		# vz = x[:, :, ACT_DIM+9] # v vertical/height
		z = x[:, :, self.INDEX]
		total_distance = z.mean(dim=1)
		return total_distance
	
	def metrics(self, x, **kwargs):
		# if x is numpy array, convert it to torch tensor
		if isinstance(x, np.ndarray): x = torch.from_numpy(x)
		with torch.no_grad():
			return {
				self.NAME: self.cal_value(x),
			}

class MujocoFaster(SingleValueGuide):
	LOWER = False
	INDEX = 14
	NAME = "speed"

class MujocoSlower(SingleValueGuide):
	LOWER = True
	INDEX = 14
	NAME = "speed"

class MujocoHigher(SingleValueGuide):
	LOWER = False
	INDEX = 6
	NAME = "height"

class MujocoLower(SingleValueGuide):
	LOWER = True
	INDEX = 6
	NAME = "height"

class CheetahFaster(MujocoFaster):
	INDEX = 14

class CheetahSlower(MujocoSlower):
	INDEX = 14

class HopperFaster(MujocoFaster):
	INDEX = 8

class HopperSlower(MujocoSlower):
	INDEX = 8

class Walker2DFaster(MujocoFaster):
	INDEX = 14

class Walker2DSlower(MujocoFaster):
	INDEX = 14


## Maze
class Maze2dTargetGuide(NoTrainGuide):
	def __init__(self, target=[0.0457, 0.0458], **kwargs):
		kwargs["target"] = target
		super().__init__(**kwargs)

	def forward(self, x, cond, t):
		"""
		cal total distance
		x: (batch_size, trace_length, 6)
			the last dim is [x, y, vx, vy, act_x, act_y]
			we only use x, y to calculate distance
		"""
		value = self.cal_value(x)
		return - value
	
	def cal_value(self, x):
		"""
		cal total distance
		x: (batch_size, trace_length, 6)
			the last dim is [act*6, root_x, root_y, root_vx, root_vy, ...]
			we only use x, y to calculate distance
		"""
		# Extract x, y coordinates
		ACT_DIM = 2
		# z = x[:, :, ACT_DIM+0] # height
		# vx = x[:, :, ACT_DIM+8] # v horizontal
		# vz = x[:, :, ACT_DIM+9] # v vertical/height
		pos_x = x[:, -1, ACT_DIM+0]
		pos_y = x[:, -1, ACT_DIM+1]
		total_distance = (pos_x - self.kwargs["target"][0]) ** 2 + (pos_y - self.kwargs["target"][1]) ** 2
		total_distance = total_distance.sqrt() # comment this would lead to l2
		return total_distance
	
	def metrics(self, x, **kwargs):
		# if x is numpy array, convert it to torch tensor
		if isinstance(x, np.ndarray): x = torch.from_numpy(x)
		with torch.no_grad():
			return {
				"TargetGap": self.cal_value(x),
			}

class Maze2dAvoidGuide(NoTrainGuide):
	def __init__(self, target=[0.0457, 0.0458], **kwargs):
		kwargs["target"] = target
		super().__init__(**kwargs)

	def forward(self, x, cond, t):
		"""
		cal total distance
		x: (batch_size, trace_length, 6)
			the last dim is [x, y, vx, vy, act_x, act_y]
			we only use x, y to calculate distance
		"""
		value = self.cal_value(x)
		return - value
	
	def cal_value(self, x):
		ACT_DIM = 2
		RADIUS = 0.3

		# Extract x, y coordinates for all time steps
		pos_x = x[:, :, ACT_DIM+0]
		pos_y = x[:, :, ACT_DIM+1]

		# Calculate squared distance from the target for all time steps
		total_distance = (pos_x - self.kwargs["target"][0]) ** 2 + (pos_y - self.kwargs["target"][1]) ** 2

		# Calculate the distance (L2 norm) for all time steps
		total_distance = total_distance.sqrt() # (B, T)

		# Only apply loss for positions within radius R of the target
		mask = (total_distance <= RADIUS ** 2)
		
		# Make sure the mask is float type
		mask = mask.float()

		# Apply the mask to compute the effective loss for all time steps
		effective_loss = total_distance * mask

		return effective_loss.mean(dim=1)
	
	def metrics(self, x, **kwargs):
		# if x is numpy array, convert it to torch tensor
		if isinstance(x, np.ndarray): x = torch.from_numpy(x)
		with torch.no_grad():
			return {
				"TargetGap": self.cal_value(x),
			}
