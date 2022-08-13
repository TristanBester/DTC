import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim,
        seq_len,
        cnn_kernel,
        cnn_stride,
        mp_kernel,
        mp_stride,
        lstm_hidden_dim,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.cnn_kernel = cnn_kernel
        self.cnn_stride = cnn_stride
        self.mp_kernel = mp_kernel
        self.mp_stride = mp_stride
        self.lstm_hidden_dim = lstm_hidden_dim

        self.cnn = nn.Conv1d(
            in_channels=self.input_dim,
            out_channels=1,
            kernel_size=self.cnn_kernel,
            stride=self.cnn_stride,
            padding=0,
            dilation=1,
        )
        self.max_pool = nn.MaxPool1d(
            kernel_size=self.mp_kernel, stride=self.mp_stride, padding=0, dilation=1
        )

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=self.lstm_hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0,
            bidirectional=True,
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.cnn(x))
        x = self.max_pool(x)

        x = x.permute(0, 2, 1)
        x, (_, _) = self.lstm(x)

        x = x[:, :, : self.lstm_hidden_dim] + x[:, :, self.lstm_hidden_dim :]
        return x
