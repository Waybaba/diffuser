import torch
import torch.nn as nn
import pdb


class ValueGuide(nn.Module):

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

class NoTrainGuide(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, cond, t):
        """
        cal total distance
        x: (batch_size, trace_length, 6)
            the last dim is [x, y, vx, vy, act_x, act_y]
            we only use x, y to calculate distance
        """
        # Extract x, y coordinates
        coord = x[:, :, :2]  # shape: (batch_size, trace_length, 2)

        # Compute differences between successive coordinates along the trace
        diff = coord[:, 1:, :] - coord[:, :-1, :]  # shape: (batch_size, trace_length-1, 2)

        # Compute squared Euclidean distance (assuming coordinates are Euclidean)
        sqdist = (diff**2).sum(dim=-1)  # shape: (batch_size, trace_length-1)

        # Compute total distance
        total_distance = sqdist.sum(dim=-1)  # shape: (batch_size,)

        return total_distance


    def gradients(self, x, *args):
        x.requires_grad_()
        y = self(x, *args)
        grad = torch.autograd.grad([y.sum()], [x])[0]
        x.detach()
        return y, grad
