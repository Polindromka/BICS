import torch.nn as nn


class Encoder(nn.Module):
    """
    Class encoder
    """

    def __init__(self, latent_size, hidden_dims=None, in_channels=7):
        """
        Initializing encoder
        :param latent_size: The size of encoder
        :param hidden_dims: the size of layers while increasing
        :param in_channels: number of EEG channels
        """
        super().__init__()
        self.latent_size = latent_size
        hidden_dims = [64, 128, 256, 512, 1024]
        modules = []
        in_channels = 7
        for h_dim in hidden_dims[:-1]:
            modules.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=h_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            in_channels = h_dim

        modules.append(
            nn.Sequential(
                nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=1),
                nn.BatchNorm1d(1024),
                nn.LeakyReLU(),
            )
        )
        modules.append(nn.Flatten())
        modules.append(nn.Linear(hidden_dims[-1] * 4, latent_size))

        self.encoder = nn.Sequential(*modules)

    def forward(self, x):
        """
        Go through Autoencoder model
        :param x: Input signal
        :return: Output "bottleneck" or embeddings - the minimal size of decomposition
        """
        embedding = self.encoder(x)
        return embedding


class Decoder(nn.Module):
    """
    Class decoder
    """

    def __init__(self, latent_size, hidden_dims=None):
        """
        Initializing decoder
        :param latent_size: The size of decoder
        :param hidden_dims: the size of layers while decreasing
        """
        super().__init__()
        self.latent_size = latent_size

        hidden_dims = [1024, 512, 256, 128, 64]  # only 32 channels for 30
        self.linear = nn.Linear(in_features=latent_size, out_features=hidden_dims[0])

        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose1d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.LeakyReLU(),
                )
            )

        modules.append(
            nn.Sequential(
                nn.ConvTranspose1d(
                    hidden_dims[-1],
                    hidden_dims[-1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.BatchNorm1d(hidden_dims[-1]),
                nn.LeakyReLU(),
                nn.Conv1d(hidden_dims[-1], out_channels=7, kernel_size=5, padding=1),
                nn.Sigmoid(),
            )
        )
        self.decoder = nn.Sequential(*modules)


    def forward(self, x):
        """
        Go through decoder
        :param x: Input signal
        :return: Recovered signal
        """
        x = self.linear(x)
        x = x.view(-1, 1024, 1)
        recovered = self.decoder(x)
        return recovered


class AutoEncoder(nn.Module):
    """
    Class Autoencoder
    """

    def __init__(self, latent_size):
        """
        Initializing autoencoder
        :param latent_size: Number of layers
        """
        super().__init__()
        self.latent_size = latent_size
        self.encoder = Encoder(latent_size)
        self.decoder = Decoder(latent_size)

    def forward(self, x):
        """
        Go through autoencoder
        :param x: Input signal
        :return: Recovered signal and embeddings (main features)
        """
        embedding = self.encoder(x)
        recovered_x = self.decoder(embedding)
        return recovered_x, embedding
