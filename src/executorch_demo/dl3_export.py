import torch
from torchvision import models
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
from pathlib import Path


def main() -> None:
    # Model export from https://github.com/meta-pytorch/executorch-examples/tree/main/dl3/android/DeepLabV3Demo
    model = models.segmentation.deeplabv3_resnet101(weights="DEFAULT").eval()
    sample_inputs = (torch.randn(1, 3, 224, 224),)

    et_program = to_edge_transform_and_lower(
        torch.export.export(model, sample_inputs),
        partitioner=[XnnpackPartitioner()],
    ).to_executorch()

    with Path.open("dl3_xnnpack_fp32.pte", "wb") as file:
        et_program.write_to_file(file)


if __name__ == "__main__":
    main()
