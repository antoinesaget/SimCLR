import torch
from numpy.random import default_rng


class TransformsSimCLR:
    """
    A stochastic data augmentation module that transforms any given data example randomly
    resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __init__(self):
        self.rng = default_rng()
        # self.train_transform = torchvision.transforms.Compose(
        #     [
        #         # torchvision.transforms.RandomResizedCrop(size=size),
        #         # torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
        #         # torchvision.transforms.RandomApply([color_jitter], p=0.8),
        #         # torchvision.transforms.RandomGrayscale(p=0.2),
        #         # torchvision.transforms.ToTensor(),
        #     ]
        # )

        # self.test_transform = torchvision.transforms.Compose(
        #     [
        #         # torchvision.transforms.Resize(size=size),
        #         # torchvision.transforms.ToTensor(),
        #     ]
        # )

    def __call__(self, x):
        # x shape is (100, 13, 60)
        a, b = self.rng.choice(100, 2, replace=False)
        a, b = x[a], x[b]

        # Add jitter to a and b
        #a = a + self.rng.normal(0, 0.4, size=a.shape)
        #b = b + self.rng.normal(0, 0.4, size=b.shape)

        a = torch.tensor(a, dtype=torch.float32)
        b = torch.tensor(b, dtype=torch.float32)

        return a, b
        # la, lb = self.rng.choice(28, 2, replace=True)
        # ya = x[a, :, :]
        # ya[:, :la] = 0
        # yb = x[b, :, :]
        # yb[:, -lb:] = 0
        # return a, b
