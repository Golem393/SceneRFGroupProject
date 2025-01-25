"""
Code taken from https://github.com/sxyu/pixel-nerf/blob/91a044bdd62aebe0ed3a5685ca37cb8a9dc8e8ee/src/model/code.py#L6
"""
import torch
import numpy as np

class RFFEncoding(torch.nn.Module):
    """
    Implementation of Random Fourier Features (RFF) encoding with random frequencies and biases.
    """

    def __init__(self, num_freqs=6, d_in=3, sigma=1.0, include_input=True):
        super().__init__()
        # instead of D we have 2D 
        self.num_freqs = 2 *num_freqs
        self.d_in = d_in
        self.include_input = include_input
        # Output dimensions:  (cos) per frequency for each input dimension
        self.d_out = d_in * (self.num_freqs) 
        if include_input:
            self.d_out += d_in

        # Randomly sample frequencies w' ~ N(0, sigma^2)
        freqs = 2 * np.pi * torch.randn(d_in, self.num_freqs) * sigma
        self.register_buffer("_freqs", freqs)
        print("freqs shape is ",freqs.shape)

        # Randomly sample biases b' ~ U[0, 2pi]
        biases = 2 * np.pi * torch.rand(self.num_freqs)
        self.register_buffer("_biases", biases)

        

    def forward(self, x):
        """
        Apply RFF encoding.
        :param x: Input tensor of shape (batch_size, d_in)
        :return: Encoded tensor of shape (batch_size, d_out)
        """
        # Compute the projection: 2 * pi * (x @ freqs) + biases
        print("x shape is ", x.shape)
        print("freqs shape is ", self._freqs.shape)
        print(self._freqs)
        print("freqs number is ", self.num_freqs)
        projected = torch.matmul(x, self._freqs) + self._biases
        print("projected shape is ", projected.shape)

        # Compute  cosine embeddings
        cos_enc = torch.sqrt(torch.tensor(2.0)) * torch.cos(projected)
      
        encoding = cos_enc.view(x.shape[0], -1)

        # Include the raw input if specified
        if self.include_input:
            encoding = torch.cat([x, encoding], dim=-1)  # Shape: (batch_size, d_out)

        return encoding
    @classmethod
    def from_conf(cls, conf, d_in=3):
        # PyHocon construction
        return cls(
            conf.get_int("num_freqs", 6),
            d_in,
         conf.get_float("sigma", 1.0),
         conf.get_bool("include_input", True),
        )
    
    