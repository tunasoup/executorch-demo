from collections.abc import Callable
from typing import ClassVar, TypeVar

from torch import nn
from torchvision.models import mobilenet_v2
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
from torchvision.models.segmentation import deeplabv3_resnet101

M = TypeVar("M", bound=nn.Module)


class ModelRegistry:
    """Decorator-based model registration."""

    _registry: ClassVar[dict[str, nn.Module]] = {}

    @classmethod
    def register(cls, name: str | None = None) -> Callable[[type[M]], type[M]]:
        def decorator(model_class: type[M]) -> type[M]:
            key = name.lower() if name is not None else model_class.__name__.lower()
            if key in cls._registry:
                msg = f"Entry '{key}' is already registered"
                raise ValueError(msg)
            cls._registry[key] = model_class
            return model_class

        return decorator

    @classmethod
    def get_entry(cls, name: str) -> nn.Module:
        entry = cls._registry.get(name)
        if entry:
            return entry()
        msg = f"Entry '{name}' is not registered."
        raise ValueError(msg)

    @classmethod
    def list_entries(cls) -> list[str]:
        return sorted(cls._registry)


@ModelRegistry.register()
class MobileNetV2:
    def __init__(self) -> nn.Module:
        self.model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).eval()

    def get_model(self) -> nn.Module:
        return self.model


@ModelRegistry.register("dl3_resnet101")
class DeepLabV3ResNet101:
    def __init__(self) -> nn.Module:
        self.model = deeplabv3_resnet101(weights="DEFAULT").eval()

    def get_model(self) -> nn.Module:
        return self.model
