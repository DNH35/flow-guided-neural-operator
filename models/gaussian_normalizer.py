import torch
class GaussianNormalizer(object):
    """
    Normalizes data to have a mean of 0 and a standard deviation of 1.
    """
    def __init__(self, x, eps=1e-5):
        super(GaussianNormalizer, self).__init__()
        self.mean = torch.mean(x)
        self.std = torch.std(x)
        self.eps = eps

    def encode(self, x):
        """Normalize the input tensor."""
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x):
        """Denormalize the input tensor back to its original scale."""
        x = (x * (self.std + self.eps)) + self.mean
        return x

    def to(self, device):
        """Move the normalizer's mean and std to the specified device."""
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self