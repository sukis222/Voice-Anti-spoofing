from torch import nn
import torch
from torch.nn import Sequential
import torchaudio


class LogMelspec(nn.Module):
    """
    Apply Mel Spectrogram to waveform, apply logarithm
    """
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
            torchaudio.transforms.FrequencyMasking(freq_mask_param=20),
            torchaudio.transforms.TimeMasking(time_mask_param=40),
        )


    def __call__(self, batch):
        x = torch.log(self.melspec(batch).clamp_(min=1e-9, max=1e9))
        if self.training:
            x = self.spec_augs(x)
        return x


class MFM(nn.Module):
    """
    Max-Featured-Map activation
    """
    def __init__(self, in_channels):
        """
        Args:
            in_channels (int): number of input features.
        """
        super().__init__()
        self.in_channels = in_channels

    def forward(self, in_data):
        """
        MFM forward method.
        Args:
            in_data (Tensor): input vector.
        Returns:
            output (dict): output dict containing logits.
        """
        out_tensor = torch.max(in_data[:, :self.in_channels//2, :, :], in_data[:, self.in_channels//2:, :, :])
        return out_tensor


class MFM1d(nn.Module):
    """
    Max-Featured-Map activation for 1d input
    """
    def __init__(self, in_channels):
        """
        Args:
            in_channels (int): number of input features.
        """
        super().__init__()
        self.in_channels = in_channels


    def forward(self, in_data):
        """
        MFM forward method.
        Args:
            in_data (Tensor): input vector.
        Returns:
            output (dict): output dict containing logits.
        """
        out_tensor = torch.max(in_data[:, :self.in_channels//2], in_data[:, self.in_channels//2:])
        return out_tensor



class Conv_block(nn.Module):
    """
    Conv2d -> MFM
    """
    def __init__(self, input_channels, output_channels, kernel_size):
        super().__init__()
        self.block = Sequential(
            nn.Conv2d(input_channels, output_channels*2, kernel_size=(kernel_size, kernel_size), stride=1, padding=1),
            MFM(output_channels*2),
        )

    def forward(self, data_object):
        return self.block(data_object)


class LCNN(nn.Module):
    """
    LCNN architecture
    """
    def __init__(self, sample_rate, n_mels, dropout):
        """
        Args:
            sample_rate (int): input vector.
            n_mels (int): number of mels in spectrogram.
            dropout (int): dropout
        """

        super().__init__()

        self.mel_spec = LogMelspec(sample_rate, n_mels)

        self.lcnn = Sequential(
            Conv_block(1, 32, 5),

            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            Conv_block(32, 32, 1),
            nn.BatchNorm2d(32),
            Conv_block(32, 48, 3),


            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.BatchNorm2d(48),

            Conv_block(48, 48, 1),
            nn.BatchNorm2d(48),
            Conv_block(48, 64, 3),

            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            Conv_block(64, 64, 1),
            nn.BatchNorm2d(64),
            Conv_block(64, 32, 3),
            nn.BatchNorm2d(32),
            Conv_block(32, 32, 1),
            nn.BatchNorm2d(32),
            Conv_block(32, 32, 3),

            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Flatten(),
            nn.Linear(13120, 160),
            MFM1d(160),
            nn.BatchNorm1d(80),
            nn.Dropout(dropout),

            nn.Linear(80, 2),
        )

    def forward(self, data_object, *args, **kwargs):
        """
        LCNN forward method.
        Args:
            data_object (Tensor): input vector.
        Returns:
            (dict): output dict containing logits.
        """
        outputs = self.lcnn(self.mel_spec(data_object).unsqueeze(dim=1))
        return {"outputs" : outputs}


