#!/usr/bin/env python
# coding: utf-8
import copy
import time

import matplotlib.pyplot as plt
import torch.multiprocessing as mp
from sklearn.manifold import TSNE

from utils import *
from utils.metrics import evaluate
from models import build_encoder
from typing import Callable, Dict, Tuple, Union, List


from servers.build import SERVER_REGISTRY

@SERVER_REGISTRY.register()
class Server():

    def __init__(self, args):
        self.args = args
        return

    def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr):
        C = len(client_ids)
        for param_key in local_weights:
            local_weights[param_key] = sum(local_weights[param_key])/C
        return local_weights


@SERVER_REGISTRY.register()
class ServerM(Server):

    @staticmethod
    def _is_expert_parameter(param_key):
        """
        In ResNet18_MoE, the inherited fc layer is Expert 0 and
        expert_heads.* are the remaining Experts.
        """
        return (
            param_key.startswith("fc.")
            or param_key.startswith("expert_heads.")
        )

    def _use_expert_only_fedacg(self):
        return (
            bool(self.args.server.get("FedACG", False))
            and bool(self.args.model.get("moe", False))
        )

    def set_momentum(self, model):

        state_dict = model.state_dict()

        if self._use_expert_only_fedacg():
            # Keep FedACG state only for Expert parameters.
            # Backbone and Router are intentionally excluded.
            momentum_keys = [
                key for key in state_dict
                if self._is_expert_parameter(key)
            ]
            if not momentum_keys:
                raise RuntimeError(
                    "MoE + FedACG was enabled, but no Expert parameters "
                    "were found. Expected keys beginning with 'fc.' or "
                    "'expert_heads.'."
                )
        else:
            # Preserve the original FedACG / FedAvgM behavior for all
            # non-MoE experiments and non-FedACG server configurations.
            momentum_keys = list(state_dict.keys())

        self.global_delta = {
            key: torch.zeros_like(state_dict[key])
            for key in momentum_keys
        }
        self.global_momentum = {
            key: torch.zeros_like(state_dict[key])
            for key in momentum_keys
        }


    @torch.no_grad()
    def FedACG_lookahead(self, model):
        sending_model_dict = copy.deepcopy(model.state_dict())

        # The momentum dictionary determines the parameter scope:
        # for MoE + FedACG it contains Expert keys only.
        for key in self.global_momentum.keys():
            sending_model_dict[key] += (
                self.args.server.momentum * self.global_momentum[key]
            )

        model.load_state_dict(sending_model_dict)
        return copy.deepcopy(model)


    def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr):
        C = len(client_ids)

        # All parameters are averaged over the complete selected-client set.
        # There is no expert activation filtering or activation-count weighting.
        for param_key in local_weights:
            local_weights[param_key] = sum(local_weights[param_key]) / C

        if self.args.server.momentum > 0:

            if not self.args.server.get('FedACG'):
                for param_key in local_weights:
                    local_weights[param_key] += (
                        self.args.server.momentum
                        * self.global_momentum[param_key]
                    )

            # Update momentum only for keys initialized in set_momentum().
            # For MoE + FedACG, these are Expert parameters only.
            # For all other configurations, this remains the original
            # full-state-dict momentum update.
            for param_key in self.global_delta:
                self.global_delta[param_key] = (
                    sum(local_deltas[param_key]) / C
                )
                self.global_momentum[param_key] = (
                    self.args.server.momentum
                    * self.global_momentum[param_key]
                    + self.global_delta[param_key]
                )

        return local_weights


@SERVER_REGISTRY.register()
class ServerAdam(Server):    

    def set_momentum(self, model):

        global_delta = copy.deepcopy(model.state_dict())
        for key in global_delta.keys():
            global_delta[key] = torch.zeros_like(global_delta[key])

        global_momentum = copy.deepcopy(model.state_dict())
        for key in global_momentum.keys():
            global_momentum[key] = torch.zeros_like(global_momentum[key])

        global_v = copy.deepcopy(model.state_dict())
        for key in global_v.keys():
            global_v[key] = torch.zeros_like(global_v[key]) + (self.args.server.tau * self.args.server.tau)

        self.global_delta = global_delta
        self.global_momentum = global_momentum
        self.global_v = global_v


    def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr):
        C = len(client_ids)
        server_lr = self.args.trainer.global_lr

        for param_key in local_deltas:
            self.global_delta[param_key] = sum(local_deltas[param_key])/C
            self.global_momentum[param_key] = self.args.server.momentum * self.global_momentum[param_key] + (1-self.args.server.momentum) * self.global_delta[param_key]
            self.global_v[param_key] = self.args.server.beta * self.global_v[param_key] + (1-self.args.server.beta) * (self.global_delta[param_key] * self.global_delta[param_key])

        for param_key in model_dict.keys():
            model_dict[param_key] += server_lr *  self.global_momentum[param_key] / ( (self.global_v[param_key]**0.5) + self.args.server.tau)

        return model_dict

@SERVER_REGISTRY.register()
class ServerDyn(Server):    

    def set_momentum(self, model):
        #global_momentum is h^t in FedDyn paper
        global_delta = copy.deepcopy(model.state_dict())
        for key in global_delta.keys():
            global_delta[key] = torch.zeros_like(global_delta[key])

        global_momentum = copy.deepcopy(model.state_dict())
        for key in global_momentum.keys():
            global_momentum[key] = torch.zeros_like(global_momentum[key])


        self.global_delta = global_delta
        self.global_momentum = global_momentum

    def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr):
        C = len(client_ids)
        for param_key in self.global_momentum:
            self.global_momentum[param_key] -= self.args.client.Dyn.alpha / self.args.trainer.num_clients * sum(local_deltas[param_key])
            local_weights[param_key] = sum(local_weights[param_key])/C - 1/self.args.client.Dyn.alpha * self.global_momentum[param_key]
        return local_weights
