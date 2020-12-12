from typing import NamedTuple
from munch import Munch

import torch
from torch.tensor import Tensor
from torch.utils.data import Dataset
from torch.utils.data import random_split, DataLoader
from pytorch_lightning import LightningDataModule
import torchaudio
import soundfile

# currently there is no stub in npvcc2016
# from npvcc2016.PyTorch.dataset.spectrogram import NpVCC2016_spec  # type: ignore


class DataLoaderPerformance(NamedTuple):
    """
    All attributes which affect performance of [torch.utils.data.DataLoader][^DataLoader] @ v1.6.0
    [^DataLoader]:https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    """

    num_workers: int
    pin_memory: bool


class ScycloneDataModule(LightningDataModule):
    def __init__(
        self,
        args,
        batch_size: int = 64,
        performance: DataLoaderPerformance = DataLoaderPerformance(4, True),
    ):
        super().__init__()
        self.batch_size = batch_size
        self.args = Munch(args)
        self._num_worker = performance.num_workers
        self._pin_memory = performance.pin_memory

    def prepare_data(self, *args, **kwargs) -> None:
        ScycloneDataset(train=True)

    def setup(self, stage=None):
        args = self.args

        with open(args.training_list, "r") as f:
            data_list_train = f.readlines()
        with open(args.validation_list, "r") as f:
            data_list_val = f.readlines()

        if stage == "fit" or stage is None:
            self.dataset_train = ScycloneDataset(data_list_train, args, train=True)
            self.dataset_val = ScycloneDataset(data_list_val, args, train=True)
            self.batch_size_val = len(self.dataset_val)
        if stage == "test" or stage is None:
            self.dataset_test = ScycloneDataset(data_list_test, args, train=False)
            self.batch_size_test = self.batch_size

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self._num_worker,
            pin_memory=self._pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size_val,
            num_workers=self._num_worker,
            pin_memory=self._pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size_test,
            num_workers=self._num_worker,
            pin_memory=self._pin_memory,
        )


def pad_last_dim(d: Tensor, length_min: int = 160) -> Tensor:
    """
    Pad last dimension with 0 if length is not enough.
    If input is longer than `length_min`, nothing happens.
    [..., L<160] => [..., L==160]
    """
    shape = d.size()
    length_d = shape[-1]
    if length_d < length_min:
        a = torch.zeros([*shape[:-1], length_min - length_d])
        return torch.cat((d, a), -1)
    else:
        return d


def slice_last_dim(d: Tensor, length: int) -> Tensor:
    """
    Slice last dimention if length is too much.
    If input is shorter than `length`, error is thrown.
    [..., L>160] => [..., L==160]
    """
    start = torch.randint(0, d.size()[-1] - (length - 1), (1,)).item()
    return torch.narrow(d, -1, start, length)


def pad_clip(d: Tensor, length: int = 160) -> Tensor:
    return slice_last_dim(pad_last_dim(d, length), length)

def get_spec(wave: Tensor, n_fft: int, hop_length: int):
    return torchaudio.transforms.Spectrogram(n_fft, n_fft, hop_length)(wave)


class ScycloneDataset(Dataset):
    def __init__(self, data_list="", args, train: bool = True):
        self.n_class = 2
        self.n_fft = args.n_fft
        self.hop_length = args.hop_length
        self.frame_length = args.frame_length
        # {jvs_path}|{jsss_path}
        self.data_list = [l[:-1].split('|') for l in data_list]

    def __getitem__(self, n: int):
        path_a, path_b = self.data_list[n]
        wave_a = self._load_tensor(path_a)
        spec_a = get_spec(wave_a, self.n_fft, self.hop_length)
        wave_b = self._load_tensor(path_b)
        spec_b = get_spec(wave_b, self.n_fft, self.hop_length)
        return pad_clip(spec_a, self.frame_length), pad_clip(spec_b, self.frame_length)
    
    def __len__(self) -> int:
        return len(self.data_list)

    def _load_tensor(self, path):
        wave, sr = soundfile.read(path)
        wave_tensor = torch.from_numpy(wave).float()
        return wave_tensor


if __name__ == "__main__":
    # test for clip
    i = torch.zeros(2, 2, 190, 200)
    o = pad_clip(i)
    print(o.size())
