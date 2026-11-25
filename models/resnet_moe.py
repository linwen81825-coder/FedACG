"""ResNet18 with a sparse Top-k Mixture-of-Experts classification head."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.build import ENCODER_REGISTRY
from models.resnet_base import ResNet18_base


@ENCODER_REGISTRY.register()
class ResNet18_MoE(ResNet18_base):
    """
    ResNet18 shared backbone with a global sparse MoE classification head.

    Expert 0 reuses the inherited ``fc`` layer. The remaining experts are
    stored in ``expert_heads``. All expert and router parameters are ordinary
    model parameters and are synchronized/aggregated by the original server
    implementation without any expert-specific client filtering.
    """

    def __init__(self, args, num_classes=10, **kwargs):
        super().__init__(args, num_classes=num_classes, **kwargs)

        model_cfg = args.model
        self.num_experts = int(model_cfg.get("num_experts", 4))
        self.top_k = int(model_cfg.get("top_k", 2))

        if self.num_experts < 2:
            raise ValueError("MoE requires at least 2 experts.")
        if not 1 <= self.top_k <= self.num_experts:
            raise ValueError(
                "top_k must be in [1, {}], got {}.".format(
                    self.num_experts, self.top_k
                )
            )

        feature_dim = self.fc.in_features
        self.router = nn.Linear(feature_dim, self.num_experts)

        # The inherited fc layer is Expert 0.
        self.expert_heads = nn.ModuleList(
            [
                nn.Linear(feature_dim, num_classes)
                for _ in range(self.num_experts - 1)
            ]
        )

    def forward(self, x: torch.Tensor, no_relu: bool = True):
        # Preserve the original ResNet feature extractor and output interface.
        results = super().forward(x, no_relu=no_relu)

        features = results["feature"]
        if self.logit_detach:
            features = features.detach()

        router_logits = self.router(features)
        router_probs = F.softmax(router_logits, dim=-1)

        # Top-k sparse routing; selected gate weights sum to one per sample.
        topk_logits, topk_indices = torch.topk(
            router_logits, k=self.top_k, dim=-1
        )
        topk_gates = F.softmax(topk_logits, dim=-1)
        sparse_gates = torch.zeros_like(router_probs).scatter(
            dim=1, index=topk_indices, src=topk_gates
        )

        # Evaluate only experts selected by at least one sample in this batch.
        # All experts remain in the model state_dict and are globally aggregated.
        expert_modules = [self.fc] + list(self.expert_heads)
        logits = features.new_zeros((features.size(0), self.fc.out_features))

        for expert_idx, expert in enumerate(expert_modules):
            selected_mask = (topk_indices == expert_idx).any(dim=1)
            selected_samples = selected_mask.nonzero(as_tuple=True)[0]

            if selected_samples.numel() == 0:
                continue

            selected_features = features.index_select(0, selected_samples)
            expert_logits = expert(selected_features)
            selected_gates = sparse_gates.index_select(
                0, selected_samples
            )[:, expert_idx]

            weighted_logits = expert_logits * selected_gates.unsqueeze(1)
            logits = logits.index_add(0, selected_samples, weighted_logits)

        # Switch-style load-balancing auxiliary loss.
        importance = router_probs.mean(dim=0)
        assignment = F.one_hot(
            topk_indices, num_classes=self.num_experts
        ).to(router_probs.dtype).mean(dim=(0, 1))
        aux_loss = self.num_experts * torch.sum(importance * assignment)

        results["logit"] = logits
        results["layer5"] = logits
        results["aux_loss"] = aux_loss
        results["expert_indices"] = topk_indices
        results["router_probs"] = router_probs
        return results
