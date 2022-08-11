import torch.nn as nn
import torch.nn.functional as F


class TAE(nn.Module):
    def __init__(
        self,
        input_dim,
        seq_len,
        cnn_kernel,
        cnn_stride,
        mp_kernel,
        mp_stride,
        lstm_hidden_dim,
        upsample_scale,
        deconv_kernel,
        deconv_stride,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.cnn_kernel = cnn_kernel
        self.cnn_stride = cnn_stride
        self.mp_kernel = mp_kernel
        self.mp_stride = mp_stride
        self.lstm_hidden_dim = lstm_hidden_dim
        self.upsample_scale = upsample_scale
        self.deconv_kernel = deconv_kernel
        self.deconv_stride = deconv_stride

        # Ensure output sequence length matches input
        self._validate_cnn_hparams()
        self._validate_mp_hyparams()
        self._validate_dcnn_hparams()

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

        self.upsample = nn.Upsample(scale_factor=self.upsample_scale)
        self.deconv_cnn = nn.ConvTranspose1d(
            in_channels=self.lstm_hidden_dim,
            out_channels=self.input_dim,
            kernel_size=self.deconv_kernel,
            stride=self.deconv_stride,
            padding=0,
        )

    def _validate_cnn_hparams(self):
        if (self.seq_len - self.cnn_kernel) / self.cnn_stride % 1 != 0:
            raise ValueError(
                "Assumption violated, output sequence length cannot be guaranteed to match input."
            )

    def _validate_mp_hyparams(self):
        q1 = (self.seq_len - self.cnn_kernel) / self.cnn_stride
        if (q1 - self.mp_kernel + 1) / self.mp_stride % 1 != 0:
            raise ValueError(
                "Assumption violated, output sequence length cannot be guaranteed to match input."
            )

    def _validate_dcnn_hparams(self):
        lhs = (self.seq_len + self.deconv_stride - self.deconv_kernel) / (
            self.upsample_scale * self.deconv_stride
        )
        rhs = (
            (self.seq_len - self.cnn_kernel) / self.cnn_stride - self.mp_kernel + 1
        ) / self.mp_stride + 1

        if lhs != rhs:
            print(f"LHS: {lhs}\tRHS: {rhs}")
            raise ValueError(
                "Invalid hyperparameter selection. Output sequence length will not match input."
            )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.cnn(x))
        x = self.max_pool(x)

        x = x.permute(0, 2, 1)
        x, (_, _) = self.lstm(x)

        x = x[:, :, : self.lstm_hidden_dim] + x[:, :, self.lstm_hidden_dim :]

        x = x.permute(0, 2, 1)
        x = self.upsample(x)

        x = self.deconv_cnn(x)
        x = x.permute(0, 2, 1)
        return x


import torch

if __name__ == "__main__":
    x = torch.tensor([[[1]] * 100, [[1]] * 100,]).to(torch.float32)

    network = TAE(
        input_dim=1,
        seq_len=100,
        cnn_kernel=10,
        cnn_stride=3,
        mp_kernel=10,
        mp_stride=3,
        lstm_hidden_dim=8,
        upsample_scale=2,
        deconv_kernel=10,
        deconv_stride=6,
    )

    print(x.shape)
    x = network(x)
    print(x.shape)
