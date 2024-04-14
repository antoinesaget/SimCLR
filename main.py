# %%
import argparse
import os
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from cuml.linear_model import LogisticRegression
from francecrops.Dataset import Dataset
from francecrops.Experiment import SingleExperiment
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from tsai.models.ResNet import ResNet  # noqa: F401

from model import load_optimizer, save_model
from simclr import SimCLR
from simclr.modules import NT_Xent
from simclr.modules.sync_batchnorm import convert_model
from simclr.modules.transformations import TransformsSimCLR
from tfencoder import TFEncoder
from utils import yaml_config_hook


def train(args, train_loader, model, criterion, optimizer, writer):
    loss_epoch = 0
    for step, ((x_i, x_j), _) in enumerate(train_loader):
        optimizer.zero_grad()
        x_i = x_i.cuda(non_blocking=True)
        x_j = x_j.cuda(non_blocking=True)

        # positive pair, with encoding
        h_i, h_j, z_i, z_j = model(x_i, x_j)

        loss = criterion(z_i, z_j)
        loss.backward()

        optimizer.step()

        if dist.is_available() and dist.is_initialized():
            loss = loss.data.clone()
            dist.all_reduce(loss.div_(dist.get_world_size()))

        # if args.nr == 0 and step % 50 == 0:
        # print(f"Step [{step}/{len(train_loader)}]\t Loss: {loss.item()}")

        if args.nr == 0:
            writer.add_scalar('Loss/train_epoch', loss.item(), args.global_step)
            args.global_step += 1

        loss_epoch += loss.item()
    return loss_epoch


def inference(loader, simclr_model):
    feature_vector = []
    labels_vector = []
    for step, (x, y) in enumerate(loader):
        x = x.cuda(non_blocking=True)

        # get encoding
        with torch.no_grad():
            h, _, z, _ = simclr_model(x, x)

        h = h.detach()

        feature_vector.extend(h.cpu().detach().numpy())
        labels_vector.extend(y.cpu().detach().numpy())

        if step % 20 == 0:
            print(f'Step [{step}/{len(loader)}]\t Computing features...')

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    print('Features shape {}'.format(feature_vector.shape))
    print('Labels shape {}'.format(labels_vector.shape))
    return feature_vector, labels_vector


def get_features(simclr_model, train_loader, test_loader):
    train_X, train_y = inference(train_loader, simclr_model)
    test_X, test_y = inference(test_loader, simclr_model)
    return train_X, train_y, test_X, test_y


def evaluate(
    train_finetune_loader, validation_finetune_loader, validation_contrast_loader, model, criterion
):
    loss = 0
    for _, ((x_i, x_j), _) in enumerate(validation_contrast_loader):
        x_i = x_i.cuda(non_blocking=True)
        x_j = x_j.cuda(non_blocking=True)

        with torch.no_grad():
            _, _, z_i, z_j = model(x_i, x_j)
            loss += criterion(z_i, z_j).item()

    ds_rep = Dataset(*get_features(model, train_finetune_loader, validation_finetune_loader))
    lr_model = LogisticRegression(max_iter=10000, tol=1e-3)
    experiment = SingleExperiment(n_repetition=10, n_parcels=200, model=lr_model, dataset=ds_rep)
    acc, majority_vote_acc = experiment.run()

    return acc, majority_vote_acc, loss / len(validation_contrast_loader)


class Francecrops(torch.utils.data.Dataset):
    def __init__(
        self,
        transform=None,
        train=False,
        test=False,
        validation=False,
        finetune=False,
        contrast=False,
        n_parcels=None,
    ):
        self.train = train
        self.test = test
        self.validation = validation
        self.finetune = finetune
        self.contrast = contrast

        dataset = Dataset.v0_7_40k_13_preprocessed_normalized_cached(flatten=False)
        if sum([train, test, validation]) > 1 or sum([train, test, validation]) == 0:
            raise ValueError('Exactly one of train, test, validation must be True')
        if sum([finetune, contrast]) > 1 or sum([finetune, contrast]) == 0:
            raise ValueError('Exactly one of finetune, contrast must be True')

        slice_end = n_parcels * 100 if n_parcels is not None else None
        data_source, target_source = (
            (dataset.x_train, dataset.y_train)
            if train
            else (dataset.x_test, dataset.y_test)
            if test
            else (dataset.x_validation, dataset.y_validation)
        )

        if contrast:
            self.data = data_source.reshape(
                -1, 100, data_source.shape[1], data_source.shape[2]
            ).transpose(0, 1, 3, 2)[:n_parcels]
        else:
            self.data = data_source.transpose(0, 2, 1)[:slice_end]
            self.targets = target_source[:slice_end]
        self.transform = transform

        print(self.data.shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        if self.transform:
            x = self.transform(x)

        if self.contrast:
            return x, 0
        else:
            y = self.targets[idx]
            return x, y


def main(gpu, args):
    rank = args.nr * args.devices + gpu

    if args.nodes > 1:
        dist.init_process_group('nccl', rank=rank, world_size=args.world_size)
        torch.cuda.set_device(gpu)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_contrast_loader = torch.utils.data.DataLoader(
        Francecrops(transform=TransformsSimCLR(), train=True, contrast=True),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )

    train_finetune_loader = torch.utils.data.DataLoader(
        Francecrops(n_parcels=2000, train=True, finetune=True),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
    )

    validation_contrast_loader = torch.utils.data.DataLoader(
        Francecrops(transform=TransformsSimCLR(), validation=True, contrast=True),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.workers,
    )

    validation_finetune_loader = torch.utils.data.DataLoader(
        Francecrops(validation=True, finetune=True),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
    )


    
    # 18-52-55
    # 8 - 8 - 512
    # 8 - 8 - 1024
    # 4 - 4 - 512 - 6 
    # 4 - 4 - 256 - 10
    # args2 = SimpleNamespace(
    #     timeseries_length=60,
    #     timeseries_n_channels=13,
    #     window_length=6,
    #     projection_depth=1024,
    #     n_attention_heads=8,
    #     dropout=0.1,
    #     n_encoder_layers=8,
    #     n_classes=20,
    # )
    # encoder = TFEncoder(args2)
    encoder = ResNet(13, 10)

    summary(encoder, input_size=(args.batch_size, 13, 60))
    n_features = encoder.fc.in_features  # get dimensions of fc layer

    # initialize model
    model = SimCLR(encoder, args.projection_dim, n_features)
    if args.reload:
        model_fp = os.path.join(args.model_path, 'checkpoint_{}.tar'.format(args.epoch_num))
        model.load_state_dict(torch.load(model_fp, map_location=args.device.type))
    model = model.to(args.device)

    # optimizer / loss
    optimizer, scheduler = load_optimizer(args, model)
    criterion = NT_Xent(args.batch_size, args.temperature, args.world_size)

    # DDP / DP
    if args.dataparallel:
        model = convert_model(model)
        model = DataParallel(model)
    else:
        if args.nodes > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DDP(model, device_ids=[gpu])

    model = model.to(args.device)

    writer = None
    if args.nr == 0:
        writer = SummaryWriter()

    args.global_step = 0
    args.current_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
        lr = optimizer.param_groups[0]['lr']
        loss_epoch = train(args, train_contrast_loader, model, criterion, optimizer, writer)

        if epoch % 10 == 0:
            model.eval()
            val_acc, val_majority_vote_acc, validation_loss = evaluate(
                train_finetune_loader,
                validation_finetune_loader,
                validation_contrast_loader,
                model,
                criterion,
            )
            model.train()
            writer.add_scalar('Accuracy/validation', val_acc, epoch)
            writer.add_scalar('Accuracy/validation_majority_vote', val_majority_vote_acc, epoch)
            writer.add_scalar('Loss/validation', validation_loss, epoch)

        if args.nr == 0 and scheduler:
            scheduler.step()

        if args.nr == 0 and epoch % 10 == 0:
            save_model(args, model, optimizer)

        if args.nr == 0:
            writer.add_scalar('Loss/train', loss_epoch / len(train_contrast_loader), epoch)
            writer.add_scalar('Misc/learning_rate', lr, epoch)
            print(
                f'Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(train_contrast_loader)}\t lr: {round(lr, 5)}'
            )
            args.current_epoch += 1

    save_model(args, model, optimizer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SimCLR')
    config = yaml_config_hook('./config/config.yaml')
    for k, v in config.items():
        parser.add_argument(f'--{k}', default=v, type=type(v))

    args = parser.parse_args()

    # Master address for distributed data parallel
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '8000'

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.num_gpus = torch.cuda.device_count()
    args.world_size = args.devices * args.nodes

    if args.nodes > 1:
        print(
            f'Training with {args.nodes} nodes, waiting until all nodes join before starting training'
        )
        mp.spawn(main, args=(args,), nprocs=args.gpus, join=True)
    else:
        main(0, args)

# %%
