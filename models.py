# Implementation inspired from https://github.com/RobertRosenbaum/Torch2PC/blob/1df7988e1c164e0aa23f075f061a0ea4ddb90cd2/TorchSeq2PC.py
# All the code here takes a nn.Sequential object and performs the E-step of some inference learning (IL) procedure

from typing import List, OrderedDict, Tuple
import torch
import torch.nn as nn
from copy import deepcopy

class SequentialIL(nn.Module):
    def __init__(
        self,
        module_list: List[nn.Module],
        method: str,
        n_iter: int,
        loss_func: nn.Module,
        gamma: float,
        init: str,
    ):
        super().__init__()
        assert method in ["strict", "exact", "fixed_pred"]
        assert init in ["normal", "pred", "zero"]
        assert isinstance(loss_func, nn.Module)

        self.module_list = nn.ModuleList(module_list)
        self.gamma = gamma
        self.init = init
        self.loss_func = loss_func
        self.n_iter = n_iter
        self.method = method
        self.n_layers = len(module_list) + 1          

    def forward(self, input):
        for module in self.module_list:
            input = module(input)
        return input

    def e_step(self, X, Y):

        # For exact local targets, we just do backprop
        if self.method == "exact":
            Y_hat = self(X)
            loss = self.loss_func(Y_hat, Y)
            loss.backward()
            return loss, None
        
        # Initialize targets and predictions
        with torch.no_grad():
            targets = [None] * self.n_layers
            targets[0] = X
            targets[-1] = Y
            for i in range(1, self.n_layers - 1):
                pred = self.module_list[i - 1](targets[i - 1])
                if self.init == "pred":
                    targets[i] = pred
                elif self.init == "normal":
                    targets[i] = torch.randn_like(pred)
                elif self.init == "zero":
                    targets[i] = torch.zeros_like(pred)
            
            pred = [None] * self.n_layers
            pred[0] = X
            for i in range(self.n_layers - 1):
                pred[i+1] = self.module_list[i](targets[i])
                pred[i+1].requires_grad = True

        # Iterative activity updates (E-step)
        eps = [None] * self.n_layers
        for i in range(self.n_iter):
            # Compute output layer epsilon by differentiation through loss
            loss = self.loss_func(pred[-1], targets[-1])
            eps[-1] = torch.autograd.grad(loss, pred[-1], retain_graph=False)[0]

            # Compute intermediate errors and update targets
            for l in reversed(range(1, self.n_layers - 1)):
                # Compute prediction error
                eps[l] = targets[l] - pred[l]

                # Combine top-down and bottom-up gradients
                td = torch.autograd.functional.vjp(
                    self.module_list[l], targets[l], eps[l + 1]
                )[1]
                bu = -eps[l]
                dtarget = bu + td

                # Update targets
                targets[l] = targets[l] + self.gamma * dtarget

                # Update predictions if applicable
                if self.method == "strict":
                    pred[l + 1] = self.module_list[l](targets[l])

        # Set weight grads (M-step)
        for l in range(self.n_layers - 1):
            for param in self.module_list[l].parameters():
                dtheta = torch.autograd.grad(
                    outputs=pred[l + 1],
                    inputs=param,
                    grad_outputs=eps[l + 1],
                    allow_unused=True,
                    retain_graph=True,
                )[0]
                param.grad = dtheta

        return loss, targets
