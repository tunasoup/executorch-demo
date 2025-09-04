from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import ClassVar, TypeVar

from torch import nn
from torchvision.models.segmentation import deeplabv3_resnet101

M = TypeVar("M", bound="BaseModel")


class BaseModel(ABC):
    def get_model(self) -> nn.Module:
        return self.model

    @property
    @abstractmethod
    def model(self) -> nn.Module:
        pass


class ModelRegistry:
    """Decorator-based model registration."""

    _registry: ClassVar[dict[str, type[M]]] = {}

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
            return entry().get_model()
        msg = f"Entry '{name}' is not registered."
        raise ValueError(msg)

    @classmethod
    def list_entries(cls) -> list[str]:
        return sorted(cls._registry)


@ModelRegistry.register("dl3_resnet101")
class DeepLabV3ResNet101(BaseModel):
    def __init__(self) -> None:
        self.model = deeplabv3_resnet101(weights="DEFAULT").eval()

    def model(self) -> nn.Module:
        return self.model
