from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable
from typing import Any, ClassVar, Self, TypeVar

import torch
from torch import nn
from torchvision.models.segmentation import (
    DeepLabV3_ResNet50_Weights,
    DeepLabV3_ResNet101_Weights,
    deeplabv3_resnet50,
    deeplabv3_resnet101,
)

M = TypeVar("M", bound="BaseModel")
CHAINABLE_METHODS = {"eval", "train", "to", "cpu", "cuda", "half", "float"}


class BaseModel(ABC):
    """An Abstract wrapper class for nn.Modules (and their subclasses).

    Attributes:
        model (type(nn.Module)): Model with a callable forward function.
    """

    def __call__(self, *args: torch.Tensor, **kwargs) -> torch.Tensor | OrderedDict:
        return self.model(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Delegate attributes and methods to the model, overwriting a class return
        for methods, so that methods can be chained.

        Args:
            name (str): Attribute or method name.

        Returns:
            Any: Class instance in case of a chainable method, otherwise the original return value.
        """
        attr = getattr(self.model, name)
        # Wrap chainable methods so they are called when followed with parentheses
        if name in CHAINABLE_METHODS and callable(attr):

            def wrapper(*args: Any, **kwargs) -> Self:
                attr(*args, **kwargs)
                return self

            return wrapper
        # Otherwise return the attribute
        return attr

    @property
    @abstractmethod
    def model(self) -> type[nn.Module]:
        """
        Returns:
            type[nn.Module]: Model that is expected to behave like nn.Module
        """

    @abstractmethod
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Preprocess an image to match an expected input of the model.

        Args:
            x (torch.Tensor): Image(s) to preprocess.

        Returns:
            torch.Tensor: Image tensor in a valid model input form.
        """


class ModelRegistry:
    """Decorator-based model registration, case insensitive."""

    _registry: ClassVar[dict[str, type[M]]] = {}

    @classmethod
    def register(cls, name: str | None = None) -> Callable[[type[M]], type[M]]:
        def decorator(model_class: type[M]) -> type[M]:
            key = name.lower() if name is not None else model_class.__name__.lower()
            if key in cls._registry:
                msg = f"Entry '{key}' is already registered."
                raise ValueError(msg)
            cls._registry[key] = model_class
            return model_class

        return decorator

    @classmethod
    def get_entry(cls, name: str) -> type[BaseModel]:
        """Get and initialize a registered entry matching the provided name.

        Args:
            name (str): Name of an entry (case-insensitive).

        Raises:
            ValueError: If the provided name does not match a registered entry.

        Returns:
            type[BaseModel]: Initialized entry matching the name.
        """
        entry = cls._registry.get(name.lower())
        if entry:
            return entry()
        msg = f"Entry '{name}' is not registered."
        raise ValueError(msg)

    @classmethod
    def list_entries(cls) -> list[str]:
        """
        Returns:
            list[str]: List of all the registered entries in alphabetical order.
        """
        return sorted(cls._registry)

    @classmethod
    def clear(cls) -> None:
        """Clear the registry. Intended for testing purposes."""
        cls._registry.clear()

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if an entry matching the name exists.

        Args:
            name (str): Name to look up (case-insensitive).

        Returns:
            bool: Whether an entry matching the name exists.
        """
        return name.lower() in cls._registry


@ModelRegistry.register("dl3_resnet101")
class DeepLabV3ResNet101(BaseModel):
    def __init__(self) -> None:
        weights = DeepLabV3_ResNet101_Weights.DEFAULT
        # TODO similar weights metadata feature as in torchvision
        self._preprocess = weights.transforms()
        self._model = deeplabv3_resnet101(weights=weights)

    @property
    def model(self) -> type[nn.Module]:
        return self._model

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        return self._preprocess(x)


@ModelRegistry.register("dl3_resnet50")
class DeepLabV3ResNet50(BaseModel):
    def __init__(self) -> None:
        weights = DeepLabV3_ResNet50_Weights.DEFAULT
        self._preprocess = weights.transforms()
        self._model = deeplabv3_resnet50(weights=weights)

    @property
    def model(self) -> type[nn.Module]:
        return self._model

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        return self._preprocess(x)
