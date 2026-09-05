from collections.abc import Mapping
import inspect
from pathlib import Path

import torch
import torch.nn as nn
import torchvision


_CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / 'checkpoints'


def _load_checkpoint(source, aliases):
    if source is None:
        return None
    if isinstance(source, Mapping):
        checkpoint = source
    else:
        path = aliases.get(source, source)
        path = Path(path).expanduser()
        if not path.is_absolute() and not path.is_file():
            path = Path(__file__).resolve().parents[1] / path
        if not path.is_file():
            raise FileNotFoundError(
                f'Pretrained checkpoint not found: {path}. '
                'Download it into checkpoints/ or set pretrained_weights to a valid path.'
            )
        load_kwargs = {'map_location': 'cpu'}
        if 'weights_only' in inspect.signature(torch.load).parameters:
            load_kwargs['weights_only'] = True
        checkpoint = torch.load(str(path), **load_kwargs)

    if isinstance(checkpoint, Mapping) and 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    if not isinstance(checkpoint, Mapping):
        raise TypeError('A pretrained checkpoint must contain a state-dict mapping')
    return checkpoint


def _load_matching_weights(module, checkpoint):
    if checkpoint is None:
        return

    target = module.state_dict()
    matched = {}
    for key, value in checkpoint.items():
        if key.startswith('module.'):
            key = key[len('module.'):]
        candidates = (key, f'backbone.{key}')
        for candidate in candidates:
            if candidate in target and hasattr(value, 'shape') and target[candidate].shape == value.shape:
                matched[candidate] = value
                break

    if not matched:
        raise RuntimeError('The pretrained checkpoint has no parameters compatible with this encoder')
    module.load_state_dict(matched, strict=False)


class Res101Encoder(nn.Module):
    """
    Resnet101 backbone from deeplabv3
    modify the 'downsample' component in layer2 and/or layer3 and/or layer4 as the vanilla Resnet
    """

    def __init__(self, replace_stride_with_dilation=None, pretrained_weights=None):
        super().__init__()
        aliases = {
            'deeplabv3': _CHECKPOINT_DIR / 'deeplabv3_resnet101_coco-586e9e4e.pth',
            'resnet101': _CHECKPOINT_DIR / 'resnet101-63fe2227.pth',
        }
        checkpoint = _load_checkpoint(pretrained_weights, aliases)

        _model = torchvision.models.resnet.resnet101(pretrained=False,
                                                     replace_stride_with_dilation=replace_stride_with_dilation)
        self.backbone = nn.ModuleDict()
        for dic, m in _model.named_children():
            self.backbone[dic] = m

        self.reduce1 = nn.Conv2d(1024, 512, kernel_size=1, bias=False)
        self.reduce2 = nn.Conv2d(2048, 512, kernel_size=1, bias=False)
        self.reduce1d = nn.Linear(in_features=1000, out_features=1, bias=True)

        self._init_weights()
        _load_matching_weights(self, checkpoint)

    def forward(self, x):
        x = self.backbone["conv1"](x)
        x = self.backbone["bn1"](x)
        x = self.backbone["relu"](x)

        x = self.backbone["maxpool"](x)
        x = self.backbone["layer1"](x)
        x = self.backbone["layer2"](x)
        x = self.backbone["layer3"](x)    
        feature = self.reduce1(x)  # (2, 512, 64, 64)
        x = self.backbone["layer4"](x) 
        # feature map -> avgpool -> fc -> single value
        t = self.backbone["avgpool"](x)
        t = torch.flatten(t, 1)
        t = self.backbone["fc"](t)
        t = self.reduce1d(t)
        return (feature, t)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class Res50Encoder(nn.Module):
    """
    Resnet50 backbone from deeplabv3
    modify the 'downsample' component in layer2 and/or layer3 and/or layer4 as the vanilla Resnet
    """

    def __init__(self, replace_stride_with_dilation=None, pretrained_weights=None):
        super().__init__()
        aliases = {
            'deeplabv3': _CHECKPOINT_DIR / 'deeplabv3_resnet50_coco-cd0a2569.pth',
            'resnet50': _CHECKPOINT_DIR / 'resnet50-19c8e357.pth',
        }
        checkpoint = _load_checkpoint(pretrained_weights, aliases)

        _model = torchvision.models.resnet.resnet50(pretrained=False,
                                                    replace_stride_with_dilation=replace_stride_with_dilation)
        self.backbone = nn.ModuleDict()
        for dic, m in _model.named_children():
            self.backbone[dic] = m

        self.reduce1 = nn.Conv2d(1024, 512, kernel_size=1, bias=False)
        self.reduce2 = nn.Conv2d(2048, 512, kernel_size=1, bias=False)
        self.reduce1d = nn.Linear(in_features=1000, out_features=1, bias=True)

        self._init_weights()
        _load_matching_weights(self, checkpoint)

    def forward(self, x):
        x = self.backbone["conv1"](x)
        x = self.backbone["bn1"](x)
        x = self.backbone["relu"](x)

        x = self.backbone["maxpool"](x)
        x = self.backbone["layer1"](x)
        x = self.backbone["layer2"](x)
        x = self.backbone["layer3"](x)
        feature = self.reduce1(x)  # (2, 512, 64, 64)
        x = self.backbone["layer4"](x)
        # feature map -> avgpool -> fc -> single value
        t = self.backbone["avgpool"](x)
        t = torch.flatten(t, 1)
        t = self.backbone["fc"](t)
        t = self.reduce1d(t)
        return (feature, t)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
