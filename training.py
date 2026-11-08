import argparse
import torch.nn.functional as F
import torch
import random
import os
from torch import tensor
from networks import CPFGNN
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_auc_score
from partition_utils import partition_index
from dataset_loader import DataLoader

try:
    import torch_geometric.data.data
    import torch_geometric.data.storage
    torch.serialization.add_safe_globals([
        torch_geometric.data.data.Data,
        torch_geometric.data.data.DataEdgeAttr,
        torch_geometric.data.data.DataTensorAttr,
        torch_geometric.data.storage.GlobalStorage,
    ])
except (ImportError, AttributeError):
    pass

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    try:
        torch.use_deterministic_algorithms(True)
    except:
        pass

def RunExp(args, dataset, data, partition_data):
    device = torch.device('cuda:'+str(args.device) if torch.cuda.is_available() else 'cpu')
    labels = data.y.to(device)
    if labels.dim() > 1:
        labels = labels.squeeze()

    model = CPFGNN(args, partition_data).to(device)
    optimizer = torch.optim.Adam([
        {'params': model.prop_g.parameters(), 
         'weight_decay': args.prop_g_wd, 'lr': args.prop_g_lr},
        {'params': model.prop_f.parameters(), 
         'weight_decay': args.prop_f_wd, 'lr': args.prop_f_lr},
        {'params': model.lin1.parameters(),
         'weight_decay': args.weight_decay, 'lr': args.lr},
        {'params': model.lin2.parameters(),
         'weight_decay': args.weight_decay, 'lr': args.lr}
    ])
    
    def train(model, optimizer, data, labels):
        model.train()
        optimizer.zero_grad()
        logits = model(partition_data)
        train_mask = data.train_mask
        if logits.shape[1] == 1:
            loss = F.binary_cross_entropy_with_logits(
                logits[train_mask].squeeze(-1), 
                labels[train_mask].float())
        else:
            loss = F.nll_loss(F.log_softmax(logits[train_mask], dim=1), labels[train_mask])
        
        loss.backward()
        optimizer.step()
        return loss.item()
    
    def test(model, data, labels):
        model.eval()
        with torch.no_grad():
            logits = model(partition_data)
            accs = []
            for mask in [data.train_mask, data.val_mask, data.test_mask]:
                if logits.shape[1] == 1:
                    pred = (logits[mask].squeeze(-1) > 0).to(torch.long)
                    acc = roc_auc_score(y_true=labels[mask].cpu().numpy(), 
                                       y_score=logits[mask].squeeze(-1).cpu().numpy())
                else:
                    pred = logits[mask].max(1)[1]
                    acc = pred.eq(labels[mask]).sum().item() / mask.sum().item()
                accs.append(acc)
            return accs
    
    best_val_acc = 0
    test_acc = 0
    patience = 0
    for epoch in range(args.epochs):
        loss = train(model, optimizer, data, labels)
        train_acc, val_acc, tmp_test_acc = test(model, data, labels)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
            patience = 0
        else:
            patience += 1
        if args.early_stopping > 0 and patience >= args.early_stopping:
            break
    
    return test_acc, best_val_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='seeds for random splits.')
    parser.add_argument('--epochs', type=int, default=1000, help='max epochs.')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay.')
    parser.add_argument('--early_stopping', type=int, default=200, help='early stopping.')
    parser.add_argument('--hidden', type=int, default=64, help='hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout for neural networks.')
    
    parser.add_argument('--train_rate', type=float, default=0.6, help='train set rate.')
    parser.add_argument('--val_rate', type=float, default=0.2, help='val set rate.')
    parser.add_argument('--K', type=int, default=10, help='propagation steps.')
    parser.add_argument('--dprate', type=float, default=0.5, help='dropout for propagation layer.')
    
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--device', type=int, default=0, help='GPU device.')
    parser.add_argument('--runs', type=int, default=5, help='number of runs.')
    
    parser.add_argument('--prop_g_lr', type=float, default=0.01, help='learning rate for structure propagation.')
    parser.add_argument('--prop_g_wd', type=float, default=0.0005, help='weight decay for structure propagation.')
    parser.add_argument('--prop_f_lr', type=float, default=0.01, help='learning rate for feature propagation.')
    parser.add_argument('--prop_f_wd', type=float, default=0.0005, help='weight decay for feature propagation.')
    parser.add_argument('--num_partitions', type=int, default=5, help='number of partitions.')
    
    args = parser.parse_args()
    
    # 10 fixed seeds for random splits (same as SGNN-LS)
    SEEDS = [1941488137, 4198936517, 983997847, 4023022221, 4019585660, 
             2108550661, 1648766618, 629014539, 3212139042, 2424918363]
    
    print(args)
    print("---------------------------------------------")

    dataset = DataLoader(args.dataset)
    print(f'Dataset: {args.dataset}, Classes: {dataset.num_classes}')
    saver = torch.load('./data_saved/'+args.dataset+'.pt', weights_only=False)
    
    data = dataset[0]
    data.edge_index = saver['edge_index']
    data.y = data.y.flatten()
    args.runs = saver['train_mask'].shape[1] if len(saver['train_mask'].shape) > 1 else 1
    args.num_features = data.x.size(1)
    args.num_classes = dataset.num_classes
    
    device = torch.device('cuda:'+str(args.device) if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print('--Graph partition precomputing.')
    set_seed(1)

    class DatasetWrapper:
        def __init__(self, pyg_data):
            self.graph = {
                'node_feat': pyg_data.x,
                'edge_index': pyg_data.edge_index,
                'num_nodes': pyg_data.x.size(0)
            }
            self.num_features = pyg_data.x.size(1)
            self.num_classes = int(pyg_data.y.max().item()) + 1
            self.label = pyg_data.y
    
    dataset_wrapper = DatasetWrapper(data)
    partition_data = partition_index(dataset_wrapper, args.num_partitions, device)
    
    print('--Preprocessing completed. GNN training begins.')
    
    results = []
    for RP in range(args.runs):
        print(f"RP {RP} Launched...")
        args.seed = SEEDS[RP]
        set_seed(args.seed)

        if len(saver['train_mask'].shape) > 1:
            data.train_mask = saver['train_mask'][:, RP].to(device)
            data.val_mask = saver['val_mask'][:, RP].to(device)
            data.test_mask = saver['test_mask'][:, RP].to(device)
        else:
            data.train_mask = saver['train_mask'].to(device)
            data.val_mask = saver['val_mask'].to(device)
            data.test_mask = saver['test_mask'].to(device)
        
        test_acc, best_val_acc = RunExp(args, dataset, data, partition_data)
        results.append([test_acc, best_val_acc])
        print(f'run_{str(RP+1)} \t test_acc: {test_acc:.4f}')
    
    test_acc_mean, val_acc_mean = np.mean(results, axis=0) * 100
    values = np.asarray(results)[:, 0]
    uncertainty = np.max(np.abs(sns.utils.ci(sns.algorithms.bootstrap(values, func=np.mean, n_boot=1000), 95) - values.mean()))
    
    print(f'CPFGNN on dataset {args.dataset}, in {args.runs} repeated experiment:')
    print(f'test acc mean = {test_acc_mean:.4f} ± {uncertainty*100:.4f}  \t val acc mean = {val_acc_mean:.4f}')
