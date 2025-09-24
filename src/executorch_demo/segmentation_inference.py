import math

import matplotlib.pyplot as plt
import torch
from torchvision.io import decode_image
from torchvision.utils import draw_segmentation_masks

from executorch_demo.models import ModelRegistry
from executorch_demo.utils import create_logger, get_data_raw_dir, get_files_with_extensions

logger = create_logger(__name__)


def plot_images(imgs: torch.Tensor | list[torch.Tensor]) -> None:
    """Plot a list of tensor images in a grid.

    Args:
        imgs (torch.Tensor | list[torch.Tensor]): 4D Tensor or a list of 3D Tensors.
    """
    if isinstance(imgs, torch.Tensor) and imgs.dim() == 4:
        imgs = [imgs[i] for i in range(imgs.size(0))]
    elif not isinstance(imgs, list):
        imgs = [imgs]

    n_imgs = len(imgs)
    n_rows, n_cols = optimal_grid(n_imgs, target_ratio=16 / 9)
    fig, axes = plt.subplots(n_rows, n_cols)
    axes = axes.flatten() if n_imgs > 1 else [axes]
    for i, img in enumerate(imgs):
        img_to_plot = img.detach().cpu().permute(1, 2, 0)
        axes[i].imshow(img_to_plot)

    # Turn off axes for all subplots, even those without an image
    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def optimal_grid(count: int, target_ratio: float = 16 / 9) -> tuple[int, int]:
    """Get the optimal grid size (rows, cols) for subplotting images.

    Args:
        count (int): Number of images.
        target_ratio (float, optional): Target aspect ratio.

    Returns:
        tuple[int, int]: Number of rows and columns.
    """
    return min(
        ((r, math.ceil(count / r)) for r in range(1, count + 1)),
        key=lambda rc: abs((rc[1] / rc[0]) - target_ratio),
    )


def run_inference() -> None:
    """Run segmentation inference with a hardcoded model and images."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Starting inference on device %s", device)

    data_path = get_data_raw_dir() / "misc"
    input_img_paths = get_files_with_extensions(data_path, {"jpg", "jpeg"}, recursive=True)
    input_imgs = [decode_image(d) for d in input_img_paths]
    # Keep only 1:1 aspect ratio
    input_imgs = [img for img in input_imgs if img.shape[1] == img.shape[2]]
    logger.info("Loaded %d images for inference", len(input_imgs))
    # plot_images(input_imgs)

    model_name = "dl3_resnet50"
    model = ModelRegistry.get_entry(model_name).eval().to(device)
    logger.info("Loaded model %s ", model_name)

    batch = torch.stack([model.preprocess(d).to(device) for d in input_imgs])
    output = model(batch)["out"]  # (n_imgs, n_classes, h, w)
    logger.info("Output shape: %s", output.shape)
    n_classes = output.shape[1]

    # Given the output, get the predicted class for each pixel (n_imgs, h, w)
    predictions = output.argmax(1)

    # Create a mask for each class (n_imgs, n_classes, h, w)
    masks = predictions.unsqueeze(1) == torch.arange(n_classes, device=predictions.device).view(
        1, -1, 1, 1
    )

    # Resize the list of input images to match the model output size
    out_h, out_w = predictions.shape[1:]
    resized_imgs = [
        torch.nn.functional.interpolate(
            img.unsqueeze(0).float(),
            size=(out_h, out_w),
            mode="bilinear",
            align_corners=False,
        )
        .squeeze(0)
        .to(torch.uint8)
        if img.shape[1:] != (out_h, out_w)
        else img
        for img in input_imgs
    ]

    # Draw the masks on the resized images, with some transparency
    segmented_imgs = [
        draw_segmentation_masks(img, mask, alpha=0.5)
        for img, mask in zip(resized_imgs, masks[:, 1:].cpu(), strict=True)
    ]

    # Stack original and segmented images for comparison, creating (h, 2w) images
    stacked_imgs = [
        torch.cat((img, seg_img), dim=2)
        for img, seg_img in zip(resized_imgs, segmented_imgs, strict=True)
    ]

    logger.info("Inference done, plotting images...")
    plot_images(stacked_imgs)
    # plot_images(input_imgs + segmented_imgs)


if __name__ == "__main__":
    run_inference()
