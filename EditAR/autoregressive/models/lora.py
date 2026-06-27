import math
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r if r > 0 else 1.0

        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs)) if bias else None
        if r > 0:
            self.lora_A = nn.Parameter(torch.empty((r, in_features), **factory_kwargs))
            self.lora_B = nn.Parameter(torch.empty((out_features, r), **factory_kwargs))
            self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)
            self.lora_dropout = nn.Identity()
        self.reset_parameters()

    @classmethod
    def from_linear(cls, linear, r=8, lora_alpha=16, lora_dropout=0.0):
        lora_linear = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        lora_linear.weight.data.copy_(linear.weight.data)
        if linear.bias is not None:
            lora_linear.bias.data.copy_(linear.bias.data)
        return lora_linear

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        if self.r > 0:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x):
        result = F.linear(x, self.weight, self.bias)
        if self.r > 0:
            lora_update = F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)
            result = result + lora_update * self.scaling
        return result


def _matches_target(module_name: str, target_modules: Sequence[str]) -> bool:
    return any(module_name == target or module_name.endswith(f".{target}") for target in target_modules)


def inject_lora(model: nn.Module, target_modules: Iterable[str], r=8, lora_alpha=16, lora_dropout=0.0) -> int:
    target_modules = tuple(target_modules)
    replaced = 0
    for module_name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full_name = f"{module_name}.{child_name}" if module_name else child_name
            if isinstance(child, nn.Linear) and _matches_target(full_name, target_modules):
                setattr(module, child_name, LoRALinear.from_linear(child, r, lora_alpha, lora_dropout))
                replaced += 1
    return replaced


def mark_only_lora_as_trainable(model: nn.Module, extra_trainable_keywords: Iterable[str] = ()) -> None:
    extra_trainable_keywords = tuple(extra_trainable_keywords)
    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name or any(keyword in name for keyword in extra_trainable_keywords)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def lora_state_dict(model: nn.Module, extra_trainable_keywords: Iterable[str] = ()) -> dict:
    extra_trainable_keywords = tuple(extra_trainable_keywords)
    return {
        name: param.detach().cpu()
        for name, param in model.state_dict().items()
        if "lora_" in name or any(keyword in name for keyword in extra_trainable_keywords)
    }
