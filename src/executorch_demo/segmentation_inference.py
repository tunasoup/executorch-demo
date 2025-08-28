import math

import matplotlib.pyplot as plt
import torch
from torchvision.io import decode_image
from torchvision.models.segmentation import FCN_ResNet50_Weights, fcn_resnet50
from torchvision.utils import draw_segmentation_masks

from executorch_demo.utils import create_logger, get_data_raw_dir

logger = create_logger(__name__)


def plot_images(imgs: torch.Tensor | list[torch.Tensor]) -> None:
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
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()


def optimal_grid(count: int, target_ratio: float = 16 / 9) -> tuple[int, int]:
    return min(
        ((r, math.ceil(count / r)) for r in range(1, count + 1)),
        key=lambda rc: abs((rc[1] / rc[0]) - target_ratio),
    )


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Starting inference on device %s", device)
    data_path = get_data_raw_dir() / "misc"
    input_img_paths = data_path.glob("*.jpg")
    input_imgs = [decode_image(d) for d in input_img_paths]

    # plot_images(input_imgs)

    weights = FCN_ResNet50_Weights.DEFAULT
    # preprocess = weights.transforms(resize_size=None)
    preprocess = weights.transforms()
    classes = weights.meta["categories"]
    n_classes = len(classes)

    model = fcn_resnet50(weights=weights, progress=False)
    model = model.eval().to(device)

    batch = torch.stack([preprocess(d).to(device) for d in input_imgs])
    output = model(batch)["out"]  # (n_imgs, n_classes, h, w)

    # Given the output , get the predicted class for each pixel (n_imgs, h, w)
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

    plot_images(stacked_imgs)
    # plot_images(input_imgs + segmented_imgs)


if __name__ == "__main__":
    main()
