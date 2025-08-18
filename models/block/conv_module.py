import warnings
from functools import partial
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def efficient_conv_bn_eval_forward(bn: nn.modules.batchnorm._BatchNorm, conv: nn.modules.conv._ConvNd, x: torch.Tensor):
    """
    Implementation based on https://arxiv.org/abs/2305.11624
    "Tune-Mode ConvBN Blocks For Efficient Transfer Learning"
    It leverages the associative law between convolution and affine transform,
    i.e., normalize (weight conv feature) = (normalize weight) conv feature.
    It works for Eval mode of ConvBN blocks during validation, and can be used
    for training as well. It reduces memory and computation cost.

    Args:
        bn (nn._BatchNorm): a BatchNorm module.
        conv (nn._ConvNd): a conv module.
        x (torch.Tensor): Input feature map.
    """
    # These lines of code are designed to deal with various cases
    # like bn without affine transform, and conv without bias
    weight_on_the_fly = conv.weight
    if conv.bias is not None:
        bias_on_the_fly = conv.bias
    else:
        bias_on_the_fly = torch.zeros_like(bn.running_var)

    if bn.weight is not None:
        bn_weight = bn.weight
    else:
        bn_weight = torch.ones_like(bn.running_var)

    if bn.bias is not None:
        bn_bias = bn.bias
    else:
        bn_bias = torch.zeros_like(bn.running_var)

    # shape of [C_out, 1, 1, 1] in Conv2d
    weight_coeff = torch.rsqrt(bn.running_var + bn.eps).reshape([-1] + [1] * (len(conv.weight.shape) - 1))
    # shape of [C_out, 1, 1, 1] in Conv2d
    coefff_on_the_fly = bn_weight.view_as(weight_coeff) * weight_coeff

    # shape of [C_out, C_in, k, k] in Conv2d
    weight_on_the_fly = weight_on_the_fly * coefff_on_the_fly
    # shape of [C_out] in Conv2d
    bias_on_the_fly = bn_bias + coefff_on_the_fly.flatten() * (bias_on_the_fly - bn.running_mean)

    return conv._conv_forward(x, weight_on_the_fly, bias_on_the_fly)


class ConvModule(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: Union[bool, str] = "auto",
        conv_cfg: Optional[Dict] = None,
        norm_cfg: Optional[Dict] = None,
        act_cfg: Optional[Dict] = dict(type="ReLU"),
        inplace: bool = True,
        with_spectral_norm: bool = False,
        padding_mode: str = "zeros",
        order: tuple = ("conv", "norm", "act"),
        efficient_conv_bn_eval: bool = False,
    ):
        super().__init__()

        # Validate inputs
        assert isinstance(order, tuple) and len(order) == 3
        assert set(order) == {"conv", "norm", "act"}
        official_padding_modes = ["zeros", "circular", "reflect"]
        if padding_mode not in official_padding_modes:
            raise ValueError(f"Padding mode {padding_mode} not supported. " f"Choose from {official_padding_modes}")

        # Configuration
        self.order = order
        self.with_explicit_padding = padding_mode == "reflect"
        self.padding_mode = padding_mode
        self.padding = padding
        self.with_spectral_norm = with_spectral_norm
        self.inplace = inplace

        # Norm layer presence
        self.with_norm = norm_cfg is not None
        if bias == "auto":
            bias = not self.with_norm
        self.with_bias = bias

        # Activation layer presence
        self.with_activation = act_cfg is not None
        self.act_cfg = act_cfg

        # Build padding layer if needed
        if self.with_explicit_padding:
            self.pad = self._get_padding_layer(padding_mode, padding)

        # Build conv layer
        conv_padding = 0 if self.with_explicit_padding else padding
        self.conv = self._build_conv_layer(
            in_channels, out_channels, kernel_size, stride, conv_padding, dilation, groups, bias, conv_cfg
        )

        # Apply spectral norm if requested
        if with_spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)

        # Build norm layer
        self.norm_name = None
        if self.with_norm:
            norm_channels = out_channels if order.index("norm") > order.index("conv") else in_channels
            self.norm = self._build_norm_layer(norm_cfg, norm_channels)
            self.norm_name = "norm"
            if self.with_bias and isinstance(self.norm, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                warnings.warn("Unnecessary conv bias before batch/instance norm")

        # Build activation layer
        if self.with_activation:
            self.activate = self._build_activation_layer(act_cfg)

        # Efficient conv-bn eval
        self.efficient_conv_bn_eval_forward = None
        self.turn_on_efficient_conv_bn_eval(efficient_conv_bn_eval)

        # Initialize weights
        self._init_weights()

    def _get_padding_layer(self, mode, padding):
        """Create padding layer for reflect padding."""
        if isinstance(padding, int):
            return lambda x: F.pad(x, (padding,) * 4, mode=mode)
        elif isinstance(padding, tuple) and len(padding) == 2:
            return lambda x: F.pad(x, (padding[1], padding[1], padding[0], padding[0]), mode=mode)
        else:
            raise ValueError(f"Invalid padding: {padding}")

    def _build_conv_layer(
        self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, conv_cfg
    ):
        """Create convolution layer with optional config."""
        conv_type = conv_cfg.get("type", "Conv2d") if conv_cfg else "Conv2d"

        if conv_type == "Conv2d":
            return nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        elif conv_type == "Conv1d":
            return nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        else:
            raise ValueError(f"Unsupported conv type: {conv_type}")

    def _build_norm_layer(self, norm_cfg, num_features):
        """Create normalization layer."""
        norm_type = norm_cfg.get("type", "BatchNorm2d")

        if norm_type == "BatchNorm2d":
            return nn.BatchNorm2d(num_features, **{k: v for k, v in norm_cfg.items() if k != "type"})
        elif norm_type == "InstanceNorm2d":
            return nn.InstanceNorm2d(num_features, **{k: v for k, v in norm_cfg.items() if k != "type"})
        else:
            raise ValueError(f"Unsupported norm type: {norm_type}")

    def _build_activation_layer(self, act_cfg):
        """Create activation layer."""
        act_type = act_cfg.get("type", "ReLU")
        kwargs = {k: v for k, v in act_cfg.items() if k != "type"}

        if act_type == "ReLU":
            return nn.ReLU(inplace=self.inplace)
        elif act_type == "LeakyReLU":
            return nn.LeakyReLU(inplace=self.inplace)
        elif act_type == "Sigmoid":
            return nn.Sigmoid()
        elif act_type == "Tanh":
            return nn.Tanh()
        elif act_type == "GELU":
            return nn.GELU()
        elif act_type == "HSigmoid":
            return nn.Hardsigmoid()
        else:
            raise ValueError(f"Unsupported activation: {act_type}")

    def _init_weights(self):
        """Initialize weights."""
        # Conv layer initialization
        if self.with_activation and self.act_cfg.get("type") == "LeakyReLU":
            nn.init.kaiming_normal_(
                self.conv.weight, a=self.act_cfg.get("negative_slope", 0.01), mode="fan_out", nonlinearity="leaky_relu"
            )
        else:
            nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")

        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)

        # Norm layer initialization
        if self.with_norm:
            nn.init.constant_(self.norm.weight, 1)
            nn.init.constant_(self.norm.bias, 0)

    def forward(self, x, activate=True, norm=True):
        for layer in self.order:
            if layer == "conv":
                if self.with_explicit_padding:
                    x = self.pad(x)

                # Check for efficient conv-bn fusion
                next_layer_idx = self.order.index(layer) + 1
                if (
                    (next_layer_idx < len(self.order))
                    and (self.order[next_layer_idx] == "norm")
                    and norm
                    and self.with_norm
                    and not self.norm.training
                    and self.efficient_conv_bn_eval_forward is not None
                ):

                    # Use fused operation
                    x = efficient_conv_bn_eval_forward(self.norm, self.conv, x)
                    # Skip the next norm layer since we fused it
                    continue
                else:
                    x = self.conv(x)

            elif layer == "norm" and norm and self.with_norm:
                x = self.norm(x)

            elif layer == "act" and activate and self.with_activation:
                x = self.activate(x)

        return x

    def turn_on_efficient_conv_bn_eval(self, enable=True):
        """Enable/disable efficient conv-bn fusion."""
        if enable and self.with_norm and isinstance(self.norm, nn.BatchNorm2d) and self.norm.track_running_stats:
            self.efficient_conv_bn_eval_forward = efficient_conv_bn_eval_forward
        else:
            self.efficient_conv_bn_eval_forward = None

    @classmethod
    def create_from_conv_bn(
        cls, conv: nn.modules.conv._ConvNd, bn: nn.modules.batchnorm._BatchNorm, efficient_conv_bn_eval=True
    ):
        """Alternative constructor from existing conv and bn layers."""
        module = cls.__new__(cls)
        super(cls, module).__init__()

        # Basic attributes
        module.order = ("conv", "norm", "act")
        module.with_norm = True
        module.with_activation = False
        module.with_bias = conv.bias is not None
        module.with_spectral_norm = False
        module.with_explicit_padding = False
        module.padding_mode = "zeros"

        # Add original layers
        module.conv = conv
        module.norm_name = "bn"
        module.add_module(module.norm_name, bn)

        # Initialize other attributes
        module.in_channels = conv.in_channels
        module.out_channels = conv.out_channels
        module.kernel_size = conv.kernel_size
        module.stride = conv.stride
        module.padding = conv.padding
        module.dilation = conv.dilation
        module.groups = conv.groups

        # Setup efficient eval
        module.turn_on_efficient_conv_bn_eval(efficient_conv_bn_eval)

        return module

    @property
    def norm(self):
        return getattr(self, self.norm_name) if self.norm_name else None
