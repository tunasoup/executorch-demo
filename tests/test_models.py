from typing import Any

import pytest

from executorch_demo.models import BaseModel, ModelRegistry


class DummyModel(BaseModel):
    def model(self) -> Any:
        pass

    def preprocess(self, x: Any) -> Any:
        pass


def test_register_model_success() -> None:
    ModelRegistry.clear()
    ModelRegistry.register()(DummyModel)

    assert ModelRegistry.is_registered(DummyModel.__name__)


def test_register_model_with_custom_name() -> None:
    ModelRegistry.clear()
    name = "custom"
    ModelRegistry.register(name)(DummyModel)

    assert ModelRegistry.is_registered(name)


def test_register_model_duplicate_name_raises() -> None:
    ModelRegistry.clear()
    ModelRegistry.register()(DummyModel)

    with pytest.raises(ValueError, match="already registered"):
        ModelRegistry.register()(DummyModel)


def test_get_entry_returns_instance() -> None:
    ModelRegistry.clear()
    ModelRegistry.register()(DummyModel)

    instance = ModelRegistry.get_entry(DummyModel.__name__)
    assert isinstance(instance, DummyModel)


def test_get_entry_unknown_name_raises() -> None:
    ModelRegistry.clear()

    with pytest.raises(ValueError, match="not registered"):
        ModelRegistry.get_entry("nonexistent")


def test_list_entries_returns_sorted_names() -> None:
    class AModel(BaseModel):
        pass

    class BModel(BaseModel):
        pass

    ModelRegistry.clear()
    ModelRegistry.register()(BModel)
    ModelRegistry.register()(AModel)

    entries = ModelRegistry.list_entries()
    assert entries == ["amodel", "bmodel"]


def test_register_model_case_insensitive() -> None:
    class AnotherModel(BaseModel):
        pass

    ModelRegistry.clear()
    ModelRegistry.register(name="Test")(DummyModel)

    with pytest.raises(ValueError, match="already registered"):
        ModelRegistry.register(name="test")(AnotherModel)


def test_is_registered() -> None:
    ModelRegistry.clear()
    ModelRegistry.register()(DummyModel)

    assert ModelRegistry.is_registered(DummyModel.__name__) is True
    assert ModelRegistry.is_registered("nonexistent") is False
    assert ModelRegistry.is_registered(DummyModel.__name__.upper()) is True
