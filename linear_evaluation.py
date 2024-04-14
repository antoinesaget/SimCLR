import argparse
import os
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import torch
from cuml.linear_model import LogisticRegression
from francecrops.Dataset import Dataset
from francecrops.Experiment import Experiment
from tsai.models.ResNet import ResNet

from simclr import SimCLR
from tfencoder import TFEncoder
from utils import yaml_config_hook


def finetune(simclr_model, train_loader, test_loader, n, n_epochs, args):
    simclr_model.finetune = True
    simclr_model.encoder.fc = torch.nn.Linear(1024, 20)
    simclr_model.to(args.device)
    optimizer = torch.optim.Adam(simclr_model.parameters(), lr=1e-4)
    loss = torch.nn.CrossEntropyLoss()

    # Get a subset of size n from the training data
    train_subset = []
    train_labels_subset = []
    for step, (x, y) in enumerate(train_loader):
        if step * x.size(0) >= n:
            break
        train_subset.append(x)
        train_labels_subset.append(y)
    train_subset = torch.cat(train_subset, dim=0)[:n]
    train_labels_subset = torch.cat(train_labels_subset, dim=0)[:n]

    for epoch in range(n_epochs):
        running_loss = 0.0
        for step in range(0, n, args.batch_size):
            optimizer.zero_grad()
            x = train_subset[step : step + args.batch_size].cuda(non_blocking=True)
            y = train_labels_subset[step : step + args.batch_size].cuda(non_blocking=True)

            h = simclr_model(x, x)
            loss_value = loss(h, y)
            loss_value.backward()
            optimizer.step()

            running_loss += loss_value.item()
            if step % 20 == 0:
                print(
                    f'Epoch {epoch+1}/{args.epochs}, Step {step}/{n}, Loss: {running_loss/20:.4f}'
                )
                running_loss = 0.0

        if epoch % 10 == 0:
            # Compute validation loss and accuracy
            val_loss = 0.0
            correct = 0
            total = 0 
            with torch.no_grad():
                for x, y in test_loader:
                    x = x.to(args.device)
                    y = y.to(args.device)
                    h = simclr_model(x, x)
                    val_loss += loss(h, y).item()
                    _, predicted = torch.max(h.data, 1)
                    total += y.size(0)
                    correct += (predicted == y).sum().item()
            val_loss /= len(test_loader)
            val_acc = correct / total

            print(
                f'Epoch {epoch+1}/{args.epochs}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}'
            )


def inference(loader, simclr_model, device):
    feature_vector = []
    labels_vector = []
    start_time = datetime.now()
    for step, (x, y) in enumerate(loader):
        x = x.to(device)

        # get encoding
        with torch.no_grad():
            h, _, z, _ = simclr_model(x, x)

        h = h.detach()

        feature_vector.extend(h.cpu().detach().numpy())
        labels_vector.extend(y.cpu().detach().numpy())

        if step % 20 == 0:
            end_time = datetime.now()
            time_per_sample = (
                (end_time - start_time).total_seconds() * 1000 * 1000 / (20 * x.size(0))
            )
            print(
                f'Step [{step}/{len(loader)}]\t Computing features... Inference speed: {time_per_sample:.2f} ms per 1000 samples'
            )
            start_time = datetime.now()

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    print('Features shape {}'.format(feature_vector.shape))
    print('Labels shape {}'.format(labels_vector.shape))
    return feature_vector, labels_vector


def get_features(simclr_model, train_loader, test_loader, device):
    train_X, train_y = inference(train_loader, simclr_model, device)
    test_X, test_y = inference(test_loader, simclr_model, device)
    return train_X, train_y, test_X, test_y


class Francecrops(torch.utils.data.Dataset):
    def __init__(self, transform=None, test=False, train=False):
        self.transform = None
        dataset = Dataset.v0_7_40k_13_preprocessed_normalized_cached(flatten=False)
        if train:
            self.data = dataset.x_train  # [: 280 * 100]
            self.targets = dataset.y_train  # [: 280 * 100]
        if test:
            self.data = dataset.x_test
            self.targets = dataset.y_test
        self.data = self.data.transpose(0, 2, 1)
        self.n_classes = len(np.unique(self.targets))
        print(self.data.shape, self.targets.shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]
        return x, y


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SimCLR')
    config = yaml_config_hook('./config/config.yaml')
    for k, v in config.items():
        parser.add_argument(f'--{k}', default=v, type=type(v))

    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {args.device}')

    train_ds = Francecrops(train=True)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.linear_evaluation_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
    )

    test_ds = Francecrops(test=True)
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=args.linear_evaluation_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
    )

    train_loader_finetune = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.linear_evaluation_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
    )

    test_loader_finetune = torch.utils.data.DataLoader(
        test_ds,
        batch_size=args.linear_evaluation_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
    )

    # args2 = SimpleNamespace(
    #     timeseries_length=60,
    #     timeseries_n_channels=13,
    #     window_length=6,
    #     projection_depth=512,
    #     n_attention_heads=4,
    #     dropout=0.1,
    #     n_encoder_layers=4,
    #     n_classes=20,
    # )
    # encoder = TFEncoder(args2)
    encoder = ResNet(train_ds.data.shape[1], 10)
    encoder.training = False
    n_features = encoder.fc.in_features  # get dimensions of fc layer

    # load pre-trained model from checkpoint
    simclr_model = SimCLR(encoder, args.projection_dim, n_features)
    model_fp = os.path.join(args.model_path, 'checkpoint_{}.tar'.format(args.epoch_num))
    simclr_model.load_state_dict(torch.load(model_fp, map_location=args.device.type))
    simclr_model = simclr_model.to(args.device)

    # finetune(simclr_model, train_loader_finetune, test_loader_finetune, 100 * 100, 100, args)

    simclr_model.eval()

    print('### Creating features from pre-trained context model ###')
    (train_X, train_y, test_X, test_y) = get_features(
        simclr_model, train_loader, test_loader, args.device
    )

    print(f'Shape of train_X: {train_X.shape}, train_y: {train_y.shape}')
    print(f'Shape of test_X: {test_X.shape}, test_y: {test_y.shape}')

    simclr_rep_dataset = Dataset(train_X, train_y, test_X, test_y)
    if args.save_rep:
        date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        simclr_rep_dataset.to_parquet(f'df_v0.7_40k_rep_simclr_all_bands_{args.epoch_num}_{date}')

    if args.linear_evaluation:
        model = LogisticRegression(max_iter=10000, tol=1e-3)
        n_trains = np.array([100, 1000, 10000]) * 100
        experiment = Experiment(n_trains, [model], [])
        experiment.set_representation(
            simclr_rep_dataset, f'SimCLR 13 Bands - {args.epoch_num} epochs'
        )
        experiment.runExperiment()
        res = experiment.getResults()
        print(res)
