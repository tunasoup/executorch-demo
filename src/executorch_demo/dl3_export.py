from pathlib import Path

import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
from torchvision import models

from executorch_demo.utils import get_model_dir


def get_dl3_model_and_inputs() -> tuple[torch.nn.Module, torch.Tensor]:
    model = models.segmentation.deeplabv3_resnet101(weights="DEFAULT").eval()
    sample_inputs = (torch.randn(1, 3, 224, 224),)
    return model, sample_inputs


def export_dl3_executorch() -> None:
    # Model export from https://github.com/meta-pytorch/executorch-examples/tree/main/dl3/android/DeepLabV3Demo
    model, sample_inputs = get_dl3_model_and_inputs()

    et_program = to_edge_transform_and_lower(
        torch.export.export(model, sample_inputs),
        partitioner=[XnnpackPartitioner()],
    ).to_executorch()

    output_file = get_model_dir() / "dl3_xnnpack_fp32.pte"

    with Path.open(output_file, "wb") as file:
        et_program.write_to_file(file)


def export_dl3_onnx() -> None:
    model, sample_inputs = get_dl3_model_and_inputs()

    output_file = get_model_dir() / "dl3_fp32.onnx"

    torch.onnx.export(
        model, sample_inputs, output_file, input_names="input", output_names="output", dynamo=True
    )


if __name__ == "__main__":
    # export_dl3_executorch()
    export_dl3_onnx()
