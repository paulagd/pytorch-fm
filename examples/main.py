import torch
import tqdm
from datetime import datetime
import os
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from IPython import embed


from torchfm.dataset.avazu import AvazuDataset
from torchfm.dataset.criteo import CriteoDataset
from torchfm.dataset.movielens import MovieLens1MDataset, MovieLens20MDataset
from torchfm.model.afi import AutomaticFeatureInteractionModel
from torchfm.model.afm import AttentionalFactorizationMachineModel
from torchfm.model.dcn import DeepCrossNetworkModel
from torchfm.model.dfm import DeepFactorizationMachineModel
from torchfm.model.ffm import FieldAwareFactorizationMachineModel
from torchfm.model.fm import FactorizationMachineModel
from torchfm.model.fnfm import FieldAwareNeuralFactorizationMachineModel
from torchfm.model.fnn import FactorizationSupportedNeuralNetworkModel
from torchfm.model.lr import LogisticRegressionModel
from torchfm.model.ncf import NeuralCollaborativeFiltering
from torchfm.model.nfm import NeuralFactorizationMachineModel
from torchfm.model.pnn import ProductNeuralNetworkModel
from torchfm.model.wd import WideAndDeepModel
from torchfm.model.xdfm import ExtremeDeepFactorizationMachineModel


def get_dataset(name, path):
    if name == 'movielens1M':
        return MovieLens1MDataset(path)
    elif name == 'movielens20M':
        return MovieLens20MDataset(path)
    elif name == 'criteo':
        return CriteoDataset(path)
    elif name == 'avazu':
        return AvazuDataset(path)
    else:
        raise ValueError('unknown dataset name: ' + name)


def get_model(name, dataset):
    """
    Hyperparameters are empirically determined, not opitmized.
    """
    field_dims = dataset.field_dims
    if name == 'lr':
        return LogisticRegressionModel(field_dims)
    elif name == 'fm':
        return FactorizationMachineModel(field_dims, embed_dim=16)
    elif name == 'ffm':
        return FieldAwareFactorizationMachineModel(field_dims, embed_dim=4)
    elif name == 'fnn':
        return FactorizationSupportedNeuralNetworkModel(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'wd':
        return WideAndDeepModel(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'ipnn':
        return ProductNeuralNetworkModel(field_dims, embed_dim=16, mlp_dims=(16,), method='inner', dropout=0.2)
    elif name == 'opnn':
        return ProductNeuralNetworkModel(field_dims, embed_dim=16, mlp_dims=(16,), method='outer', dropout=0.2)
    elif name == 'dcn':
        return DeepCrossNetworkModel(field_dims, embed_dim=16, num_layers=3, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'nfm':
        return NeuralFactorizationMachineModel(field_dims, embed_dim=64, mlp_dims=(64,), dropouts=(0.2, 0.2))
    elif name == 'ncf':
        # only supports MovieLens dataset because for other datasets user/item colums are indistinguishable
        assert isinstance(dataset, MovieLens20MDataset) or isinstance(dataset, MovieLens1MDataset)
        return NeuralCollaborativeFiltering(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2,
                                            user_field_idx=dataset.user_field_idx,
                                            item_field_idx=dataset.item_field_idx)
    elif name == 'fnfm':
        return FieldAwareNeuralFactorizationMachineModel(field_dims, embed_dim=4, mlp_dims=(64,), dropouts=(0.2, 0.2))
    elif name == 'dfm':
        return DeepFactorizationMachineModel(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'xdfm':
        return ExtremeDeepFactorizationMachineModel(
            field_dims, embed_dim=16, cross_layer_sizes=(16, 16), split_half=False, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'afm':
        return AttentionalFactorizationMachineModel(field_dims, embed_dim=16, attn_size=16, dropouts=(0.2, 0.2))
    elif name == 'afi':
        return AutomaticFeatureInteractionModel(
            field_dims, embed_dim=32, num_heads=4, num_layers=2, mlp_dims=(16, 16), dropouts=(0.2, 0.2))
    else:
        raise ValueError('unknown model name: ' + name)


def train(model, optimizer, data_loader, criterion, device, writer, epoch, log_interval=1000):
    model.train()
    total_loss = 0
    for i, (fields, target) in enumerate(tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)):
        embed()
        fields, target = fields.to(device), target.to(device)
        y = model(fields)
        loss = criterion(y, target.float())
        model.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar('loss/train', loss.item(), epoch)
        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            print('    - loss:', total_loss / log_interval)
            total_loss = 0


def test(model, data_loader, device, writer, epoch=None, criterion=None):
    model.eval()
    targets, predicts = list(), list()
    val_loss = None
    with torch.no_grad():
        for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            fields, target = fields.to(device), target.to(device)
            y = model(fields)
            if epoch is not None and criterion is not None:
                val_loss = criterion(y, target.float())
                writer.add_scalar('loss/val', val_loss.item(), epoch)
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
    return val_loss, roc_auc_score(targets, predicts)


def main(dataset_name,
         dataset_path,
         model_name,
         epoch,
         learning_rate,
         batch_size,
         weight_decay,
         device,
         save_dir,
         writer,
         no_chkp):

    device = torch.device(device)
    dataset = get_dataset(dataset_name, dataset_path)
    train_length = int(len(dataset) * 0.8)
    valid_length = int(len(dataset) * 0.1)
    test_length = len(dataset) - train_length - valid_length
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, (train_length, valid_length, test_length))
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=8)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8)
    model = get_model(model_name, dataset).to(device)
    os.makedirs(save_dir, exist_ok=True)
    criterion = torch.nn.BCELoss()
    # IDEA: loss = - (prediction_i - prediction_j).sigmoid().log().sum()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_val_loss = np.inf
    best_epoch, best_auc = 0, 0

    for epoch_i in range(epoch):
        train(model, optimizer, train_data_loader, criterion, device, writer, epoch_i)
        val_loss, auc = test(model, valid_data_loader, device, writer, epoch_i, criterion)
        writer.add_scalar(f'validation_metrics/AUC', auc, epoch_i)
        # EARLY STOPPING
        if val_loss < best_val_loss:
            best_auc = auc
            best_epoch = epoch_i
            best_val_loss = val_loss
            if not no_chkp:
                torch.save(model, f'{save_dir}/{model_name}_{epoch_i}.pt')
        # print('epoch:', epoch_i, 'validation: auc:', auc)
        print(f'epoch:{epoch_i}/{epoch}: Validation AUC:{auc}')
    _, auc = test(model, test_data_loader, device, writer, None)
    print('test auc:', auc)
    print("-------------------")
    print(f'BEST EPOCH: {best_epoch} : AUC = {best_auc}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='movielens1M', help='movielens1M, movielens20M, criteo, avazu')
    parser.add_argument('--dataset_path', default="../datasets/ml-1m/ratings.dat", help='criteo/train.txt, avazu/train, or ml-1m/ratings.dat')
    parser.add_argument('--model_name', default='fm')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save_dir', default='chkpt')
    parser.add_argument("--no_chkp", action="store_true", default=False, help="Disables saving checkpoints when true")
    parser.add_argument("--logs", action="store_true", default=False, help="Enables logs")
    parser.add_argument("--logsname", default="", help="Enables logs")
    args = parser.parse_args()

    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # for visualization
    date = datetime.now().strftime('%y%m%d%H%M%S')
    if args.logs:
        if len(args.logsname) == 0:
            writer = SummaryWriter(log_dir=f'logs/logs_{date}_{args.model_name}/')
        else:
            writer = SummaryWriter(log_dir=f'logs/logs_{args.logsname}_{args.model_name}/')
    else:
        writer = SummaryWriter(log_dir=f'logs/nologs/logs/')

    main(args.dataset_name,
         args.dataset_path,
         args.model_name,
         args.epoch,
         args.learning_rate,
         args.batch_size,
         args.weight_decay,
         args.device,
         args.save_dir,
         writer,
         args.no_chkp)
