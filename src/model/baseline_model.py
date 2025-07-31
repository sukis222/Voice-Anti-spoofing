from torch import nn
import torch
from torch.nn import Sequential
import torchaudio


class LogMelspec(nn.Module):
    def __init__(self, sample_rate, n_mels):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.melspec = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=400,
                win_length=400,
                hop_length=160,
                n_mels=self.n_mels
        )

        self.spec_augs = nn.Sequential(
                torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
                torchaudio.transforms.TimeMasking(time_mask_param=35),
        )

    def __call__(self, batch):
        x = torch.log(self.melspec(batch).clamp_(min=1e-9, max=1e9))
        if self.training:  # self.training - это флаг, устанавливаемый .train() и .eval()
            x = self.spec_augs(x)
        return x


class MFM(nn.Module):
    """
    Max-Featured-Map activation
    """
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        if self.in_channels % 2 != 0:
            raise ValueError("amount of channels for MFM 2/1 must be an even number.")

    def forward(self, in_features):
        out_tensor = torch.max(in_features[:, :self.in_channels//2, :, :], in_features[:, self.in_channels//2:, :, :])
        return out_tensor


class Conv_block(nn.Module):
    def __init__(self, input_channels, output_channels, dropout):
        super().__init__()
        self.block = Sequential(
            nn.Conv2d(input_channels, output_channels*2, kernel_size=(3,3), stride=1, padding=1),
            MFM(output_channels*2),
            nn.BatchNorm2d(output_channels),
            nn.Dropout(dropout)
        )

    def forward(self, data_object):
        return self.block(data_object)


class LCNN(nn.Module):
    """
    LCNN
    """
    def __init__(self, input_channels, hidden_channels, output_size, flatten_size, sample_rate, n_mels, dropout):
        super().__init__()

        self.mel_spec = LogMelspec(sample_rate, n_mels)

        self.lcnn = Sequential(
            Conv_block(input_channels, hidden_channels, dropout),
            Conv_block(hidden_channels, hidden_channels, dropout),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            Conv_block(hidden_channels, hidden_channels, dropout),
            Conv_block(hidden_channels, hidden_channels, dropout),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            Conv_block(hidden_channels, hidden_channels, dropout),
            Conv_block(hidden_channels, hidden_channels, dropout),
            nn.Flatten(),
            nn.Linear(flatten_size, output_size),
            #Узнаем на пробном прогоне
        )

    def forward(self, data_object, *args, **kwargs):
        outputs = self.lcnn(self.mel_spec(data_object).unsqueeze(dim=1))
        return {"outputs" : outputs}


