from torch import Tensor
import torch
import torch.nn as nn

class ScycloneWaveRNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.ch_base = args.base_channels
        self.ch_hidden = args.hidden_channels
        self.n_bins = args.n_bins
        self.n_frames = args.n_frames
        self.seq_len = self.n_bins * self.n_frames
        self.module_01 = nn.Sequential(
            nn.Linear(self.seq_len, self.ch_base),
            nn.ReLU(),
            nn.Linear(self.ch_base, self.ch_base * 2),
            nn.ReLU(),
            nn.Linear(self.ch_base * 2, self.ch_base * 4),
            nn.ReLU(),
            nn.Linear(self.ch_base * 4, self.ch_base * 8),
            nn.ReLU(),
        )
        self.gru = nn.GRUCell(self.ch_hidden + 1, self.ch_base)
        self.module_02 = nn.Sequential(
            nn.Linear(self.ch_base, self.ch_base),
            nn.ReLU(),
            nn.Linear(self.ch_base, 2),
        )
        self.device = args.device

    def forward(self, x, device):
        # (batch, seq_len, n_bins, n_frames)
        batch, seq_len, _, _ = x.shape
        # (batch, seq_len, n_bins * n_frames)
        x = x.view(batch, seq_len, self.n_bins * self.n_frames)
        # (batch, seq_len, ch_base * 8)
        x = self.module_01(x)
        # (batch, seq_len * n_bins, ch_hidden)
        x = x.view(batch, seq_len * self.n_bins, self.ch_hidden)
        # (seq_len * n_bins, batch, ch_hidden)
        x = x.permute(1, 0, 2)
        # (n_bins * seq_len) * (batch, ch_hidden)
        x_seq = x.split(1)

        h = torch.zeros((batch, self.ch_base), device=device)
        m_seq = []
        s_seq = []
        samples = []
        m = torch.zeros((batch,), device=device)
        s = torch.zeros((batch,), device=device)
        prev_sample = torch.zeros((batch, 1), device=device)
        
        for x in x_seq:
            # (batch, ch_hidden) -> (batch, ch_hidden + 1)
            x = torch.cat([x.squeeze(0), prev_sample], dim=1)
            # (batch, ch_hidden + 1) -> (batch, ch_base)
            h = self.gru(x, h)
            # (batch, ch_base) -> (batch, 2)
            x = self.module_02(h)
            # (batch, 2) -> (2, batch)
            x = x.permute(1, 0)
            m, s = x.split(1)
            prev_sample = m + torch.exp(s) * torch.randn(batch, device=device)
            prev_sample = prev_sample.permute(1, 0)
            m_seq.append(m)
            s_seq.append(s)
            samples.append(prev_sample)

        # (batch, n_bins * seq_len)
        return (torch.cat(m_seq).permute(1, 0),
                torch.cat(s_seq).permute(1, 0),
                torch.cat(samples).permute(1, 0))
