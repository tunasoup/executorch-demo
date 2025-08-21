from pathlib import Path

import torch
from executorch import version
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge
from executorch.runtime import Runtime
from torchvision.models import mobilenet_v2
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights


def main() -> None:
    print("Hello from executorch-demo!")
    run_mobilenet_demo()


def run_mobilenet_demo() -> None:
    # Run demo from https://github.com/meta-pytorch/executorch-examples/tree/main/dl3/android/DeepLabV3Demo
    print(version.__version__)
    print(torch.__version__)

    runtime = Runtime.get()

    operator_names = runtime.operator_registry.operator_names

    print(len(operator_names))

    print(next(iter(operator_names)))

    mv2 = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)  # This is torch.nn.Module

    model = mv2.eval()  # turn into evaluation mode

    example_inputs = (torch.randn((1, 3, 224, 224)),)  # Necessary for exporting the model

    exported_graph = torch.export.export(model, example_inputs)  # Core Aten graph

    edge = to_edge(exported_graph)  # Edge Dialect

    edge_delegated = edge.to_backend(
        XnnpackPartitioner()
    )  # Parts of the graph are delegated to XNNPACK

    executorch_program = edge_delegated.to_executorch()  # ExecuTorch program
    pte_path = "mv2_xnnpack.pte"

    with Path.open(pte_path, "wb") as file:
        executorch_program.write_to_file(file)  # Serializing into .pte file

    program = runtime.load_program(pte_path)
    method = program.load_method("forward")

    t = torch.randn((1, 3, 224, 224))

    output = method.execute([t])
    assert len(output) == 1, f"Unexpected output length {len(output)}"
    assert output[0].size() == torch.Size([1, 1000]), f"Unexpected output size {output[0].size()}"
    print("PASS")
