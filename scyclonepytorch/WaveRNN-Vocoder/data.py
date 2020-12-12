import torch
from torch.tensor import Tensor
from torch.utils.data import Dataset
from torch.utils.data import random_split, DataLoader
from pytorch_lightning import LightningDataModule

import torchaudio
import soundfile

class ScycloneWaveRNNDataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
    
    def setup(self, stage=None):
        args = self.args

        with open(args.training_list, "r") as f:
            data_list_train = f.read().splitlines()
        with open(args.validation_list, "r") as f:
            data_list_val = f.read().splitlines()

        if stage == "fit" or stage is None:
            self.dataset_train = ScycloneWaveRNNDataset(
                args, data_list_train, args.train_seq_len train=True)
            self.dataset_val = ScycloneWaveRNNDataset(
                args, data_list_val, args.val_seq_len, train=True)
            self.batch_size_val = len(self.dataset_val)
        if stage == "test" or stage is None:
            self.dataset_test = ScycloneWaveRNNDataset(
                args, data_list_test, train=False)
            self.batch_size_test = self.batch_size
    
    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=self.args.batch_size_val,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory
        )
    
    def test_dataloader(self):
        return None

def crop_spec_and_wave(wave: Tensor, spec: Tensor, seq_len: int, n_frames: int):
    n_bins, total_len = spec.size()
    start = torch.randint(0, total_len - (seq_len - 1), (1,)).item()
    cropped_spec = []
    cropped_wave = torch.narrow(wave, -1, start * n_bins + (n_frames * n_bins) // 2, seq_len * n_bins)
    for i in range(seq_len):
        cropped_spec.append(
            torch.narrow(spec, -1, start + i, n_frames)
        )
    cropped_spec = torch.cat(cropped_spec)
    return cropped_wave, cropped_spec

def get_spec(wave: Tensor, n_fft: int, hop_length: int):
    return torchaudio.transforms.Spectrogram(n_fft, n_fft, hop_length)(wave)

class ScycloneWaveRNNDataset(Dataset):
    def __init__(self, args, data_list="", seq_len=8, train: bool = True):
        self.args = args
        self.data_list = data_list
        self.seq_len
    
    def __getitem__(self, n):
        path = self.data_list[n]
        wave = self._load_tensor(path).to("cpu")
        spec = get_spec(wave, self.args.n_fft, self.args.hop_length)
        cropped = crop_spec_and_wave(wave, spec, self.seq_len, self.args.n_frames)
        del wave
        del spec
        return cropped
    
    def __len__(self) -> int:
        return len(self.data_list)
    
    def _load_tensor(self, path):
        wave, sr = soundfile.read(path)
        wave_tensor = torch.from_numpy(wave).float()
        return wave_tensor

    
