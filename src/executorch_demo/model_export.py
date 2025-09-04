from pathlib import Path
from typing import Literal

import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
from torch import nn

from executorch_demo.models import ModelRegistry
from executorch_demo.utils import create_logger, get_model_dir

logger = create_logger(__name__)


def export_model(name: str, export_format: Literal["torch", "onnx"]) -> None:
    """Export a model to edge format.

    Args:
        name (str): Model to export from the model registry.
        export_type (Literal[&quot;torch&quot;, &quot;onnx&quot;]): Format of the export.

    Raises:
        ValueError: If the export format is not supported.
    """
    logger.info("Starting export for model %s in format %s", name, export_format)
    model = ModelRegistry.get_entry(name.lower())
    # Assume the same input for all models for now
    sample_inputs = (torch.randn(1, 3, 224, 224),)
    args = [model, sample_inputs, name]
    match export_format:
        case "torch":
            export_model_torch(*args)
        case "onnx":
            export_model_onnx(*args)
        case _:
            msg = f"Unknown export type {export_format}"
            raise ValueError(msg)


def export_model_torch(model: nn.Module, sample_inputs: torch.Tensor, name: str) -> None:
    output_file = (get_model_dir() / name).with_suffix(".pte")
    et_program = to_edge_transform_and_lower(
        torch.export.export(model, sample_inputs),
        partitioner=[XnnpackPartitioner()],
    ).to_executorch()

    with Path.open(output_file, "wb") as file:
        et_program.write_to_file(file)

    logger.info("Exported model to %s", output_file)


def export_model_onnx(model: nn.Module, sample_inputs: torch.Tensor, name: str) -> None:
    output_file = (get_model_dir() / name).with_suffix(".onnx")
    torch.onnx.export(
        model, sample_inputs, output_file, input_names="input", output_names="output", dynamo=True
    )
    logger.info("Exported model to %s", output_file)


if __name__ == "__main__":
    export_model("dl3_resnet101", "torch")
