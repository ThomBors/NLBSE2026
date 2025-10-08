#!/usr/bin/env python

################################################################
# Adaptation of code from: https://github.com/Cranial-XIX/FAMO #
################################################################

import copy
import random
from abc import abstractmethod
from typing import Dict, List, Tuple, Union
from collections import deque

import numpy as np

# import cvxpy as cp
import torch
import torch.nn.functional as F
from scipy.optimize import minimize


EPS = 1e-8  # for numerical stability


class WeightMethod:
    def __init__(self, n_tasks: int, device: torch.device, max_norm=1.0):
        super().__init__()
        self.n_tasks = n_tasks
        self.device = device
        self.max_norm = max_norm

    @abstractmethod
    def get_weighted_loss(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor],
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ],
        last_shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor],
        representation: Union[torch.nn.parameter.Parameter, torch.Tensor],
        **kwargs,
    ):
        pass

    def backward(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        last_shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        representation: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[Union[torch.Tensor, None], Union[dict, None]]:
        """

        Parameters
        ----------
        losses :
        shared_parameters :
        task_specific_parameters :
        last_shared_parameters : parameters of last shared layer/block
        representation : shared representation
        kwargs :

        Returns
        -------
        Loss, extra outputs
        """
        loss, extra_outputs = self.get_weighted_loss(
            losses=losses,
            shared_parameters=shared_parameters,
            task_specific_parameters=task_specific_parameters,
            last_shared_parameters=last_shared_parameters,
            representation=representation,
            **kwargs,
        )

        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(shared_parameters, self.max_norm)

        loss.backward()
        return loss, extra_outputs

    def __call__(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        **kwargs,
    ):
        return self.backward(
            losses=losses,
            shared_parameters=shared_parameters,
            task_specific_parameters=task_specific_parameters,
            **kwargs,
        )

    def parameters(self) -> List[torch.Tensor]:
        """return learnable parameters"""
        return []


class LinearScalarization(WeightMethod):
    """Linear scalarization baseline L = sum_j w_j * l_j where l_j is the loss for task j and w_h"""

    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        task_weights: Union[List[float], torch.Tensor] = None,
    ):
        super().__init__(n_tasks, device=device)
        if task_weights is None:
            task_weights = torch.ones((n_tasks,))
        if not isinstance(task_weights, torch.Tensor):
            task_weights = torch.tensor(task_weights)
        assert len(task_weights) == n_tasks
        self.task_weights = task_weights.to(device)

    def get_weighted_loss(self, losses, **kwargs):
        loss = torch.sum(losses * self.task_weights)
        return loss, dict(weights=self.task_weights)


class STL(WeightMethod):
    """Single task learning"""

    def __init__(self, n_tasks, device: torch.device, main_task):
        super().__init__(n_tasks, device=device)
        self.main_task = main_task
        self.weights = torch.zeros(n_tasks, device=device)
        self.weights[main_task] = 1.0

    def get_weighted_loss(self, losses: torch.Tensor, **kwargs):
        if len(losses) == 1:
            return losses, dict(weights=torch.tensor([1.0], device=losses.device))

        assert len(losses) == self.n_tasks
        loss = losses[self.main_task]
        return loss, dict(weights=self.weights)



class SMGS(WeightMethod):
    """Similarity Momentum Gradient Surgery"""

    def __init__(
        self, n_tasks, device: torch.device, momentum=0.9, beta2=0.99, gamma=0.1
    ):
        """

        Parameters
        ----------
        n_tasks :
        iteration_window : 'iteration' loss is averaged over the last 'iteration_window' losses
        temp :
        """
        super().__init__(n_tasks, device=device)
        self.b1 = momentum
        self.b2 = beta2
        self.m = 0
        self.v = 0
        self.t = 0
        self.gamma = gamma

    def get_weighted_loss(self, losses, shared_parameters, **kwargs):
        grad_dims = []
        for param in shared_parameters:
            grad_dims.append(param.data.numel())
        grads = torch.Tensor(sum(grad_dims), self.n_tasks).to(self.device)

        for i in range(self.n_tasks):
            if i < self.n_tasks:
                losses[i].backward(retain_graph=True)
            else:
                losses[i].backward()
            self.grad2vec(shared_parameters, grads, grad_dims, i)
            # multi_task_model.zero_grad_shared_modules()
            for p in shared_parameters:
                p.grad = None

        g, aligned_grads, weights = self.balance_magnitude(grads)
        self.overwrite_grad(shared_parameters, g, grad_dims)
        # self.update(loss)
        return losses.sum(), {
            "loss": losses,
            "grad": g,
            "method": weights,
            "weights": aligned_grads,
        }

    def balance_magnitude(self, grads):
        self.t += 1
        w = grads.clone().detach()

        mg = torch.linalg.norm(w, dim=0)
        mg_matrix = mg.unsqueeze(1)  # Shape: (N, 1)
        mg_matrix_T = mg_matrix.T  # Shape: (1, N)

        # Compute pairwise similarities using broadcasting
        similarities_matrix = (
            2 * (mg_matrix @ mg_matrix_T) / (mg_matrix**2 + mg_matrix_T**2)
        )

        mask = torch.triu(
            torch.ones_like(similarities_matrix, dtype=torch.bool), diagonal=1
        )
        similarities = similarities_matrix[mask]

        # https://arxiv.org/pdf/2010.07468v5
        self.m = self.b1 * self.m + (1 - self.b1) * w
        self.v = (
            self.b2 * self.v + ((1 - self.b2) * (1 - similarities.mean()) ** 2) + 1e-8
        )
        mhat = self.m / (1 - self.b1**self.t)
        vhat = self.v / (1 - self.b2**self.t)

        if similarities.mean() < self.gamma:
            #### l2 standardization ####
            l2_norms = torch.norm(w, dim=0, p=2)

            # Rescale each vector to have unit L2 norm
            scaled_w_l2 = w / l2_norms

            # Using the mean magnitude of the columns
            scaling_factor = l2_norms.mean()
            adjusted_g_l2 = scaled_w_l2 * scaling_factor

            # Sum of L2-normalized vectors
            g = adjusted_g_l2.sum(1)
            aligned_w = adjusted_g_l2
            w = g

        else:
            g = (w * abs(mhat) / (torch.sqrt(vhat))).sum(1)
            aligned_w = w * abs(mhat) / (torch.sqrt(vhat))
            w = g

        return g, aligned_w, w

    @staticmethod
    def grad2vec(shared_params, grads, grad_dims, task):
        # store the gradients
        grads[:, task].fill_(0.0)
        cnt = 0
        # for mm in m.shared_modules():
        #     for p in mm.parameters():

        for param in shared_params:
            grad = param.grad
            if grad is not None:
                grad_cur = grad.data.detach().clone()
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[: cnt + 1])
                grads[beg:en, task].copy_(grad_cur.data.view(-1))
            cnt += 1

    def overwrite_grad(self, shared_parameters, newgrad, grad_dims):
        newgrad = newgrad * self.n_tasks  # to match the sum loss
        cnt = 0

        # for mm in m.shared_modules():
        #     for param in mm.parameters():
        for param in shared_parameters:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[: cnt + 1])
            this_grad = newgrad[beg:en].contiguous().view(param.data.size())
            param.grad = this_grad.data.clone()
            cnt += 1

    def backward(
        self,
        losses: torch.Tensor,
        parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        **kwargs,
    ):
        g, w = self.get_weighted_loss(losses, shared_parameters)
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(shared_parameters, self.max_norm)
        return None, {
            "GTG": g,
            "weights": w,
        }  # NOTE: to align with all other weight methods




class WeightMethods:
    def __init__(self, method: str, n_tasks: int, device: torch.device, **kwargs):
        """
        :param method:
        """
        assert method in list(METHODS.keys()), f"unknown method {method}."

        self.method = METHODS[method](n_tasks=n_tasks, device=device, **kwargs)

    def get_weighted_loss(self, losses, **kwargs):
        return self.method.get_weighted_loss(losses, **kwargs)

    def backward(
        self, losses, **kwargs
    ) -> Tuple[Union[torch.Tensor, None], Union[Dict, None]]:
        return self.method.backward(losses, **kwargs)

    def __ceil__(self, losses, **kwargs):
        return self.backward(losses, **kwargs)

    def parameters(self):
        return self.method.parameters()


METHODS = dict(
    stl=STL,
    ls=LinearScalarization,
    smgs=SMGS,
)
