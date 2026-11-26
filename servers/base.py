#!/usr/bin/env python
# coding: utf-8
import copy
import time

import matplotlib.pyplot as plt
import torch
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

    @staticmethod
    def _expert_index(param_key):
        """Map MoE classifier state-dict keys to expert ids (fc is Expert 0)."""
        if param_key.startswith("fc."):
            return 0
        if param_key.startswith("expert_heads."):
            parts = param_key.split(".")
            if len(parts) >= 3:
                try:
                    return int(parts[1]) + 1
                except ValueError:
                    return None
        return None

    @staticmethod
    def _active_client_indices(expert_idx, expert_usages, num_clients):
        if expert_usages is None or len(expert_usages) != num_clients:
            raise ValueError(
                "MoE aggregation requires one expert-usage vector for each local client result."
            )

        active = []
        for client_pos, usage in enumerate(expert_usages):
            if usage is None:
                raise ValueError(
                    "A participating MoE client returned no expert-usage statistics."
                )
            usage = torch.as_tensor(usage)
            if usage.numel() <= expert_idx:
                raise ValueError(
                    "Expert-usage vector is shorter than the requested expert index."
                )
            if int(usage[expert_idx].item()) > 0:
                active.append(client_pos)
        return active

    @staticmethod
    def _mean(values, indices=None):
        if indices is None:
            selected = values
        else:
            selected = [values[i] for i in indices]
        if not selected:
            raise ValueError("Cannot average an empty client list.")
        return sum(selected) / float(len(selected))

    def aggregate(
        self,
        local_weights,
        local_deltas,
        client_ids,
        model_dict,
        current_lr,
        expert_usages=None
    ):
        num_clients = len(client_ids)
        is_moe = bool(self.args.model.get('moe', False))

        for param_key, values in local_weights.items():
            expert_idx = self._expert_index(param_key) if is_moe else None

            if expert_idx is None:
                # Backbone, Router, and all other shared state: equal mean
                # over every client selected in this communication round.
                local_weights[param_key] = self._mean(values)
                continue

            active = self._active_client_indices(
                expert_idx, expert_usages, num_clients
            )
            if active:
                # Every client that used this expert receives equal weight,
                # independent of its number of routed samples/tokens.
                local_weights[param_key] = self._mean(values, active)
            else:
                # No client trained this expert: leave its global value intact.
                local_weights[param_key] = model_dict[param_key].clone()

        return local_weights


@SERVER_REGISTRY.register()
class ServerM(Server):

    def _is_moe_fedacg(self):
        return (
            bool(self.args.model.get('moe', False))
            and bool(self.args.server.get('FedACG', False))
        )

    def set_momentum(self, model):
        state_dict = model.state_dict()

        if self._is_moe_fedacg():
            # Independent momentum tensors are maintained only for Experts.
            momentum_keys = [
                key for key in state_dict
                if self._expert_index(key) is not None
            ]
        else:
            # Preserve the original behavior for non-MoE FedACG/FedAvgM runs.
            momentum_keys = list(state_dict.keys())

        self.global_delta = {
            key: torch.zeros_like(state_dict[key]) for key in momentum_keys
        }
        self.global_momentum = {
            key: torch.zeros_like(state_dict[key]) for key in momentum_keys
        }

    @torch.no_grad()
    def FedACG_lookahead(self, model):
        sending_model_dict = copy.deepcopy(model.state_dict())
        # For MoE FedACG, global_momentum contains Expert keys only.
        # For the original non-MoE path, it still contains every state key.
        for key in self.global_momentum.keys():
            sending_model_dict[key] += (
                self.args.server.momentum * self.global_momentum[key]
            )

        model.load_state_dict(sending_model_dict)
        return copy.deepcopy(model)

    def aggregate(
        self,
        local_weights,
        local_deltas,
        client_ids,
        model_dict,
        current_lr,
        expert_usages=None
    ):
        num_clients = len(client_ids)
        momentum = self.args.server.momentum
        is_moe_fedacg = self._is_moe_fedacg()

        if is_moe_fedacg:
            # Backbone and Router: ordinary equal average over all selected clients.
            # Experts: expert-specific active-client average + independent FedACG.
            for param_key, values in local_weights.items():
                expert_idx = self._expert_index(param_key)

                if expert_idx is None:
                    local_weights[param_key] = self._mean(values)
                    continue

                active = self._active_client_indices(
                    expert_idx, expert_usages, num_clients
                )

                if not active:
                    # No active client: delta is zero, momentum only decays,
                    # and the actual global expert parameter does not move.
                    self.global_delta[param_key] = torch.zeros_like(
                        self.global_delta[param_key]
                    )
                    self.global_momentum[param_key] = (
                        momentum * self.global_momentum[param_key]
                    )
                    local_weights[param_key] = model_dict[param_key].clone()
                    continue

                # local_deltas are measured from the expert-only lookahead
                # initialization: theta_{e,t-1} + lambda * m_{e,t-1}.
                delta_e = self._mean(local_deltas[param_key], active)
                self.global_delta[param_key] = delta_e
                self.global_momentum[param_key] = (
                    momentum * self.global_momentum[param_key] + delta_e
                )

                # theta_{e,t} = theta_{e,t-1} + m_{e,t}
                # model_dict is the pre-lookahead global model for this round.
                local_weights[param_key] = (
                    model_dict[param_key] + self.global_momentum[param_key]
                )

            return local_weights

        # Original ServerM behavior for non-MoE FedACG and FedAvgM.
        # When MoE is used with a non-FedACG ServerM variant, experts still
        # average only over clients that activated them; momentum behavior
        # remains the original ServerM behavior for that selected algorithm.
        for param_key, values in local_weights.items():
            expert_idx = (
                self._expert_index(param_key)
                if bool(self.args.model.get('moe', False))
                else None
            )
            if expert_idx is None:
                local_weights[param_key] = self._mean(values)
            else:
                active = self._active_client_indices(
                    expert_idx, expert_usages, num_clients
                )
                local_weights[param_key] = (
                    self._mean(values, active)
                    if active else model_dict[param_key].clone()
                )

        if momentum > 0:
            if not self.args.server.get('FedACG'):
                for param_key in local_weights:
                    local_weights[param_key] += momentum * self.global_momentum[param_key]

            for param_key in local_deltas:
                expert_idx = (
                    self._expert_index(param_key)
                    if bool(self.args.model.get('moe', False))
                    else None
                )
                if expert_idx is None:
                    delta = self._mean(local_deltas[param_key])
                else:
                    active = self._active_client_indices(
                        expert_idx, expert_usages, num_clients
                    )
                    delta = (
                        self._mean(local_deltas[param_key], active)
                        if active else torch.zeros_like(self.global_delta[param_key])
                    )
                self.global_delta[param_key] = delta
                self.global_momentum[param_key] = (
                    momentum * self.global_momentum[param_key] + delta
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
            global_v[key] = torch.zeros_like(global_v[key]) + (
                self.args.server.tau * self.args.server.tau
            )

        self.global_delta = global_delta
        self.global_momentum = global_momentum
        self.global_v = global_v

    def aggregate(
        self, local_weights, local_deltas, client_ids, model_dict, current_lr,
        expert_usages=None
    ):
        C = len(client_ids)
        server_lr = self.args.trainer.global_lr

        for param_key in local_deltas:
            self.global_delta[param_key] = sum(local_deltas[param_key]) / C
            self.global_momentum[param_key] = (
                self.args.server.momentum * self.global_momentum[param_key]
                + (1 - self.args.server.momentum) * self.global_delta[param_key]
            )
            self.global_v[param_key] = (
                self.args.server.beta * self.global_v[param_key]
                + (1 - self.args.server.beta)
                * (self.global_delta[param_key] * self.global_delta[param_key])
            )

        for param_key in model_dict.keys():
            model_dict[param_key] += (
                server_lr * self.global_momentum[param_key]
                / ((self.global_v[param_key] ** 0.5) + self.args.server.tau)
            )

        return model_dict


@SERVER_REGISTRY.register()
class ServerDyn(Server):

    def set_momentum(self, model):
        # global_momentum is h^t in the FedDyn paper.
        global_delta = copy.deepcopy(model.state_dict())
        for key in global_delta.keys():
            global_delta[key] = torch.zeros_like(global_delta[key])

        global_momentum = copy.deepcopy(model.state_dict())
        for key in global_momentum.keys():
            global_momentum[key] = torch.zeros_like(global_momentum[key])

        self.global_delta = global_delta
        self.global_momentum = global_momentum

    def aggregate(
        self, local_weights, local_deltas, client_ids, model_dict, current_lr,
        expert_usages=None
    ):
        C = len(client_ids)
        for param_key in self.global_momentum:
            self.global_momentum[param_key] -= (
                self.args.client.Dyn.alpha / self.args.trainer.num_clients
            ) * sum(local_deltas[param_key])
            local_weights[param_key] = (
                sum(local_weights[param_key]) / C
                - 1 / self.args.client.Dyn.alpha * self.global_momentum[param_key]
            )
        return local_weights
