import torch

class SimCLR(torch.nn.Module):
    """SSL model adapted from https://github.com/Spijkervet/SimCLR/blob/master/simclr/simclr.py. This extends SimCLR with a classification head to predict the original order of video frames after being randomly shuffled."""
    def __init__(self, encoder, projection_dim, n_features):
        super(SimCLR, self).__init__()

        self.encoder = encoder
        self.n_features = n_features

        # Replace the fc layer with an Identity function
        self.encoder.fc = torch.nn.Identity()

        # Classification head for frame re-ordering classification (after permuting video frames, predict original order)
        self.reordering_head = torch.nn.Linear(self.n_features, 24)

        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = torch.nn.Sequential(
            torch.nn.Linear(self.n_features, self.n_features // 2, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(self.n_features // 2, projection_dim, bias=False),
        )

    def forward(self, x_i, x_j):
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)

        t_i = self.reordering_head(h_i)
        t_j = self.reordering_head(h_j)

        z_i = self.projector(h_i)
        z_j = self.projector(h_j)

        return h_i, h_j, z_i, z_j, t_i, t_j