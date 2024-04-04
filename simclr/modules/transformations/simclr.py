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
        # print(x.shape)
        # x shape is (100, 4, 60)
        # pick a and b from 0 to 100
        a, b = self.rng.choice(100, 2, replace=False)
        return x[a], x[b]
        # la, lb = self.rng.choice(28, 2, replace=True)
        # ya = x[a, :, :]
        # ya[:, :la] = 0
        # yb = x[b, :, :]
        # yb[:, -lb:] = 0
        # return a, b
