import torch.nn as nn


class Decoder(nn.Module):
    def __init__(
        self,
        seq_len,
        cnn_kernel,
        cnn_stride,
        mp_kernel,
        mp_stride,
        upsample_scale,
        input_dim,
        hidden_dim,
        deconv_kernel,
        deconv_stride,
    ) -> None:
        super().__init__()

        self.seq_len = seq_len
        self.cnn_kernel = cnn_kernel
        self.cnn_stride = cnn_stride
        self.mp_kernel = mp_kernel
        self.mp_stride = mp_stride
        self.upsample_scale = upsample_scale
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.deconv_kernel = deconv_kernel
        self.deconv_stride = deconv_stride

        self.upsample = nn.Upsample(scale_factor=self.upsample_scale)
        self.deconv_cnn = nn.ConvTranspose1d(
            in_channels=self.hidden_dim,
            out_channels=self.input_dim,
            kernel_size=self.deconv_kernel,
            stride=self.deconv_stride,
            padding=0,
        )

        self._validate_dcnn_hparams()

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
        x = self.upsample(x)

        x = self.deconv_cnn(x)
        x = x.permute(0, 2, 1)
        return x

