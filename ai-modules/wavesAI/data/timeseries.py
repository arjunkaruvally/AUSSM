from typing import Union, List, Callable, Dict

from einops import rearrange
from icecream import ic

import lightning as L
import torch
from aeon.datasets import load_classification
from rich import print
from torch.utils.data import DataLoader

class Timeseries(L.LightningDataModule):
    def __init__(self, dataset_name: str,
        train_prop: float = 0.7,
        val_prop: float = 0.15,
        path: str = ".",
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = True,
        seed: int = 0):

        super().__init__()
        self.validation_set = None
        self.train_set = None
        self.dataset_test = None
        self.dataset_name = dataset_name
        self.train_prop = train_prop
        self.val_prop = val_prop
        self.path = path
        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        X, y, meta_data = load_classification(self.dataset_name,
                                              extract_path=self.path,
                                              return_metadata=True)

        print(meta_data)
        self.data_size, self.input_dim, self.sequence_length = X.shape
        self.num_classes = len(meta_data["class_values"])
        self.vocab_idx_to_token = meta_data["class_values"]
        self.vocab_token_to_idx = { val: key for key, val in enumerate(self.vocab_idx_to_token) }

        self.train_mean = 0
        self.train_std = 0

        self.save_hyperparameters(ignore=['_class_path'])


    def process_dataset(self):
        X, y, meta_data = load_classification(self.dataset_name,
                                              extract_path=self.path,
                                              return_metadata=True)

        # import termplotlib as tpl

        # fig = tpl.figure()
        # fig.plot(range(X[0, 0, :].shape[0]), X[0, 0, :])
        # fig.show()

        X = rearrange(X, "b d l -> b l d")  # this is the form required by the model

        X = torch.FloatTensor(X)

        self.train_mean = torch.mean(rearrange(X, "b l d -> (b l) d"), dim=0).view((1, 1, self.input_dim))
        self.train_std = torch.std(rearrange(X, "b l d -> (b l) d"), dim=0).view((1, 1, self.input_dim))

        X = (X - self.train_mean) / self.train_std

        return torch.utils.data.TensorDataset(X,
                                           torch.Tensor(list(map(lambda v: self.vocab_token_to_idx[v], y))))


    def setup(self, stage: str=None) -> None:
        full_set = self.process_dataset()

        num_train_samples = int(self.train_prop * self.data_size)
        num_validation_samples = int(self.val_prop * self.data_size)
        num_test_samples = self.data_size - (num_train_samples + num_validation_samples)

        self.dataset_train, self.dataset_val, self.dataset_test = torch.utils.data.random_split(full_set,
                                                                             [ num_train_samples, num_validation_samples, num_test_samples ],
                                                                             generator=torch.Generator().manual_seed(self.seed))

        def collate_batch(batch):
            x = torch.stack([ arr[0] for arr in batch ])
            y = torch.stack([ arr[1] for arr in batch ]).long()

            return x, y

        self._collate_fn = collate_batch

    def decode(self, y_label):
        return self.vocab_idx_to_token[y_label]

    def train_dataloader(self):
        return DataLoader(
            dataset=self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.dataset_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.dataset_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn
        )



if __name__ == "__main__":
    data_root = Path(os.environ["DATA_ROOT"])
    obj = Timeseries("SelfRegulationSCP1",
                     0.8,
                     data_root / "UEA")

    obj.setup()

    train_loader = obj.train_dataloader()

    batch = next(iter(train_loader))
    print(batch)
    x, y = batch

    print(x.shape, y.shape)

    import termplotlib as tpl

    fig = tpl.figure()
    fig.plot(range(x[0, :, 0].shape[0]), x[0, :, 0])
    fig.show()

