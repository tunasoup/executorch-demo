from pathlib import Path

import lightning as pl
import PIL.Image
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm

from executorch_demo.utils import (
    create_logger,
    get_data_preprocessed_dir,
    get_data_raw_dir,
)

logger = create_logger(__name__)


class DataModuleCityscapes(pl.LightningDataModule):
    def __init__(
        self, batch_size: int = 32, fractions: tuple[float, float, float] = (1, 1, 1)
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size

        self.dl_path = "Chris1/cityscapes"
        self.cache_dir: Path = get_data_raw_dir()
        self.data_dir: Path = get_data_preprocessed_dir() + "Cityscapes"
        self.fractions = fractions
        self.train_set = None
        self.val_set = None
        self.test_set = None

    def prepare_data(self) -> None:
        # Download the dataset if not cached
        load_dataset(self.dl_path, cache_dir=self.cache_dir)

    def setup(self, stage: str) -> None:
        if stage == "fit":
            train_path = self.check_and_preprocess_split("train")
            val_path = self.check_and_preprocess_split("validation")
            # TODO optionally lazy preprocessing?

            # TODO create Dataset class, allow storing to memory, apply fractions here
            self.train_set = ...
            # split=f"validation[:{self.fractions[1]*100}%]")

        if stage == "test":
            test_path = self.check_and_preprocess_split("test")
            self.test_set = ...

    def check_and_preprocess_split(self, split: str) -> Path:
        split_path = self.data_dir / split
        if not split_path.is_dir():
            preprocess_split(
                self.dl_path, split=split, cache_dir=self.cache_dir, data_dir=self.data_dir
            )
        return split_path

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_set, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_set, batch_size=self.batch_size)

    def state_dict(self) -> dict:
        return {"current_train_batch_index": self.current_train_batch_index}

    def load_state_dict(self, state_dict: dict) -> None:
        self.current_train_batch_index = state_dict["current_train_batch_index"]


def preprocess_and_save(el: dict, dir_out: Path, idx: int, n_digits: int) -> None:
    processed_el = {
        "image": preprocess_img(el["image"]),
        "mask": preprocess_img(el["semantic_segmentation"], is_mask=True),
    }
    path_out = dir_out / f"sample_{str(idx).zfill(n_digits)}.pt"
    torch.save(processed_el, path_out)


def preprocess_split(ds_path: str, split: str, cache_dir: Path, data_dir: Path) -> None:
    ds = load_dataset(ds_path, cache_dir=cache_dir, split=split).with_format("torch")
    ds_len = len(ds)
    n_digits = len(str(ds_len))
    dir_out = data_dir / split

    for idx, el in enumerate(tqdm(ds)):
        preprocess_and_save(el, dir_out, idx, n_digits)

    # TODO get tqdm to work for multiprocessing
    # with mp.Pool() as pool:
    #     list(tqdm(
    #         pool.starmap(
    #             preprocess_and_save,
    #             ((el, do, idx, digits) for idx, el in enumerate(ds))
    #         ),
    #         total=ds_len,
    #         desc="Preprocessing"
    #     ))

    # dataset.map does not support uint8, converting them to int64, consuming too much memory
    # ds = ds.map(
    #     lambda x: {
    #         "semantic_segmentation": preprocess(x["semantic_segmentation"], is_mask=True),
    #         "image": preprocess(x["image"], is_mask=False),
    #     },
    #     num_proc=4,
    # )
    # ds = ds.rename_column("semantic_segmentation", "mask")
    # ds.save_to_disk(do, num_proc=4)


def preprocess_img(img: torch.Tensor | PIL.Image.Image, is_mask: bool = False) -> torch.Tensor:
    # TODO if possible, add these to a model, and have the property as datamodule argument,
    #  and has the name so that models with same preprocessing can reuse the data
    transforms = [
        v2.PILToTensor(),
        # TODO allow choosing the size
        v2.Resize((224, 224)),
    ]

    if is_mask:
        transforms += [
            # Assume 3 dimensional, with equal color channel values
            v2.Lambda(lambda x: x[0].unsqueeze(0)),
            v2.ToDtype(torch.uint8, scale=False),
        ]
    else:
        transforms += [
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

    pipeline = v2.Compose(transforms)
    return pipeline(img)
