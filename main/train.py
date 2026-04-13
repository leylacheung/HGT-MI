import warnings
import deepchem as dc
import random
import numpy as np
import torch
import argparse
import mlflow
import torch_geometric
import seaborn as sns
from dataloader import  get_dataset_het
from model import HeteroTransformer
from copy import deepcopy
import  torch_geometric.transforms as T 
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, OneCycleLR
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    torch_geometric.seed_everything(seed)
    
def setup_mlflow(exp_name='DA_Interaction_Ours_fullexp_attn'):
    mlflow.set_tracking_uri('file:///data/leylazhang_proj/D-A_Interaction/Interaction_atten/mlruns')
    try:
        mlflow.create_experiment(exp_name)
    except:
        print("Experiment has been created")
    mlflow.set_experiment(exp_name)
# mlflow ui --backend-store-uri \file:///data/leylazhang_proj/D-A_Interaction/Interaction_atten/mlruns \--port 6006
        
def train(args, filename=None):
    setup_mlflow()
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.log_params(vars(args))
        if filename:
            mlflow.log_artifact( filename, artifact_path='code')     
        # if args.gpu >= 0:
        #     device = torch.device('cuda:%d' % args.gpu)
        # else:
        #     device = torch.device('cpu')
        if args.gpu >= 0:
            torch.cuda.set_device(args.gpu)          # ← # Important: set the default GPU device first
            device = torch.device('cuda')            # # It is also acceptable to keep using 'cuda'
        else:
            device = torch.device('cpu')
            
        maes, mapes, mses = [], [], []
        best_vals = []
    
        num_motif_edge_types = 102
        
        dataloader_train,  dataloader_test, dataloader_val, transformer, meta = get_dataset_het(args,transform=None)
        num_classes = meta['num_classes']
        # meta['num_classes'] = 1
        n_train = len(dataloader_train.dataset)
        n_val = len(dataloader_val.dataset)
        n_test = len(dataloader_test.dataset) 
        # print(n_train, n_val, n_test)
        
        for trial in range(args.num_trial):
            setup_seed(trial)  
            # Model initialization      
            model = HeteroTransformer(dataloader_train.dataset[0].metadata(), num_classes, args.hidden_dim, args.num_layer, heads=args.heads, conv=args.model, motif_conv=args.motif_conv, pool=args.pool, norm = 'BatchNorm' 
                        if args.bn else args.norm, transformer_norm=args.transformer_norm, l2norm=args.l2norm, dropout=args.dropout, attn_dropout=args.attn_dropout, criterion = args.criterion, jk=args.jk, final_jk = args.final_jk, aggr=args.aggr, 
                        normalize=args.normalize, first_residual=args.first_residual, residual=args.residual, motif_init=args.motif_init, cat_pe=args.cat_pe, use_bias=args.use_bias, 
                        combine_edge=args.combine_edge, root_weight=args.root_weight, num_motif_edge_types=num_motif_edge_types, 
                        clip_attn=args.clip_attn, model='Transformer').to(device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            
            if args.scheduler.startswith('step'):
                step_size, gamma = args.scheduler.split('-')[1:]
                scheduler = StepLR(optimizer, step_size=int(step_size), gamma=float(gamma))
            elif args.scheduler == 'cosine':
                scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epoch) 
            elif args.scheduler.startswith('onecycle'):
                pct_start = float(args.scheduler.split('-')[1]) if '-' in args.scheduler else 0.1
                scheduler = OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(dataloader_train), epochs=args.num_epoch, pct_start=pct_start)
            else:
                scheduler = None
                
            # Training 
            best_val = float("Inf")
            best_epoch = 0
            for epoch in range(1, args.num_epoch + 1):
                model.train()
                loss_all = 0
                for data in dataloader_train:
                    data = data.to(device)
                    optimizer.zero_grad()
                    loss = model.calc_loss(data)
                    loss_all += loss.item() * data.num_graphs
                    loss.backward()
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                mlflow.log_metric('Loss', loss_all / n_train, step=epoch)
                
                # Validation
                model.eval()
                loss_all_val = 0.0                
                with torch.no_grad():            
                    for data in dataloader_val:
                        data = data.to(device)
                        loss = model.calc_loss(data)
                        loss_all_val += loss.item() * data.num_graphs
                mlflow.log_metric('Loss_val', loss_all_val / n_val, step=epoch)
                if loss_all_val < best_val:
                    best_val = loss_all_val
                    best_model = deepcopy(model.state_dict())
                    best_epoch = epoch  

                if epoch % args.eval_freq == 0:
                    model.eval()
                    y_true = []
                    y_preds = []                   
                    with torch.no_grad():
                        for data in dataloader_test:
                            data = data.to(device)
                            y_true = y_true + data.y.cpu().reshape(-1, num_classes).tolist()
                            y_preds = y_preds + model.predict_score(data).cpu().reshape(-1, num_classes).tolist()
                    y_true = torch.Tensor(y_true)
                    y_preds = torch.Tensor(y_preds)
                    test_mask = y_true != 0
                    y_true = y_true[test_mask].reshape(-1,1).tolist()
                    y_preds = y_preds[test_mask].reshape(-1,1).tolist()                           
                    if args.normalize:
                        y_true= transformer.inverse_transform(y_true)
                        y_preds = transformer.inverse_transform(y_preds) 
                    mae = mean_absolute_error(y_true, y_preds)
                    mape = mean_absolute_percentage_error(y_true, y_preds)
                    mse = mean_squared_error(y_true, y_preds)
                    mlflow.log_metrics({'MAE_epoch': mae, 'MAPE_epoch': mape, 'MSE_epoch': mse}, step=epoch)  
                                                      
            # Test on best validation
            model.load_state_dict(best_model) 
            model.eval()
            y_true = []
            y_preds = []
            with torch.no_grad():     
                for data in dataloader_test:
                    data = data.to(device)
                    y_true = y_true + data.y.cpu().reshape(-1, num_classes).tolist()
                    y_preds = y_preds + model.predict_score(data).cpu().reshape(-1, num_classes).tolist()
            assert len(y_true) == n_test and len(y_preds) == n_test
            y_true = torch.Tensor(y_true)
            y_preds = torch.Tensor(y_preds)
            test_mask = y_true != 0
            y_true = y_true[test_mask].reshape(-1,1).tolist()
            y_preds = y_preds[test_mask].reshape(-1,1).tolist()            
            if args.normalize:
                y_true= transformer.inverse_transform(y_true)
                y_preds = transformer.inverse_transform(y_preds)                     
            mae = mean_absolute_error(y_true, y_preds)
            mape = mean_absolute_percentage_error(y_true, y_preds)
            mse = mean_squared_error(y_true, y_preds)     
            mlflow.log_metrics({'MAE_trial': mae, 'MAPE_trial': mape, 'MSE_trial': mse}, step=trial)  
            maes.append(mae)
            mapes.append(mape)
            mses.append(mse)
            mlflow.log_metric(f'Best_epoch', best_epoch, step=trial)
            mlflow.log_metric(f'Best_Val', best_val, step=trial)
            best_vals.append(best_val)

        # Log average results
        avg_val = np.mean(best_vals)
        std_val = np.std(best_vals)
        mlflow.log_metric(f'Best_Val_mean', avg_val)
        mlflow.log_metric(f'Best_Val_std', std_val)        
        
        avg_val = np.mean(maes)
        std_val = np.std(maes)
        mlflow.log_metric(f'MAE_mean', avg_val)
        mlflow.log_metric(f'MAE_std', std_val)      
        
        avg_val = np.mean(mapes)
        std_val = np.std(mapes)
        mlflow.log_metric(f'MAPE_mean', avg_val)
        mlflow.log_metric(f'MAPE_std', std_val)
        
        avg_val = np.mean(mses)
        std_val = np.std(mses)
        mlflow.log_metric(f'MSE_mean', avg_val)
        mlflow.log_metric(f'MSE_std', std_val)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument('-dataset', type=str, default='DA_Pair_51K', choices=['DA_Pair_1.2K','DA_Pair_51K'])
    parser.add_argument('-dataset_version', type=str, default='V1')
    parser.add_argument('-normalize', type=bool, default=True)
    parser.add_argument('-scaler', type=str, default='standard', choices=['minmax', 'standard'])
    parser.add_argument('-frac_train', type=float, default=0.8)
    parser.add_argument('-target_mode', type=str, default='single')
    parser.add_argument('-target_task', type=int, default=0, help= 'pce')
    parser.add_argument('-splitter', type=str, default='random')
    parser.add_argument('-model', type=str, default='GINE')
    parser.add_argument('-motif_conv', type=str, default='Transformer')
    
    parser.add_argument('-num_trial', type=int, default=5)
    parser.add_argument('-gpu', type=int, default=2)

    
    # Training
    parser.add_argument('-num_epoch', type=int, default=100)
    parser.add_argument('-eval_freq', type=int, default=10)
    parser.add_argument('-batch_size', type=int, default=8)
    parser.add_argument('-bn', type=bool, default=False)
    parser.add_argument('-norm', type=str, default=None, choices=[None, 'BatchNorm', 'LayerNorm'])
    parser.add_argument('-transformer_norm', type=str, default='LayerNorm', choices=[None, 'BatchNorm', 'LayerNorm'])
    parser.add_argument('-lr', type=float, default=0.001)
    parser.add_argument('-weight_decay', type=float, default=5e-4)
    parser.add_argument('-dropout', type=float, default=0.0)
    parser.add_argument('-attn_dropout', type=float, default=0.0)
    parser.add_argument('-criterion', type=str, default='MAE')
    parser.add_argument('-scheduler', type=str, default='onecycle-0.05')

    parser.add_argument('-num_layer', type=int, default=4)
    parser.add_argument('-hidden_dim', type=int, default=128)
    parser.add_argument('-heads', type=int, default=4)
    parser.add_argument('-l2norm', type=bool, default=False)
    parser.add_argument('-pool', type=str, default='add')
    parser.add_argument('-jk', type=str, default='cat')
    parser.add_argument('-final_jk', type=str, default='cat')
    parser.add_argument('-aggr', type=str, default='cat')
    parser.add_argument('-motif_init', type=str, default='atom_deepset') # atom_deepset
    parser.add_argument('-first_residual', type=bool, default=True)
    parser.add_argument('-residual', type=bool, default=True)
    parser.add_argument('-use_bias', type=bool, default=False)
    parser.add_argument('-cat_pe', type=bool, default=False)
    parser.add_argument('-transform', type=str, default=None, choices=[None, 'VirtualNode'])
    parser.add_argument('-best_val', type=bool, default=True)

    parser.add_argument('-root_weight', type=bool, default=True)
    parser.add_argument('-combine_edge', type=str, default='cat', choices=['add', 'add_lin', 'cat','add_lin'])
    parser.add_argument('-clip_attn', type=bool, default=True)

    args = parser.parse_args()

    for target_task in [0]:
        parser.set_defaults(target_task=target_task)
        for motif_init in ['random', 'atom_deepset']:# atom_deepset # random
            parser.set_defaults(motif_init=motif_init)
            for final_jk in ['attention']: # attention
                parser.set_defaults(final_jk=final_jk)
                for aggr in ['cat', 'add']:
                    parser.set_defaults(aggr=aggr)
                    for dataset in ['DA_Pair_51K']: # DA_Pair_1.3K, DA_Pair_51K
                        parser.set_defaults(dataset=dataset) 
                        for dim in [128,256,512]:# 256,512,1024
                            parser.set_defaults(hidden_dim=dim) 
                            for num_layer in [2,4,6,8,10]:# ,6,8,10
                                parser.set_defaults(num_layer=num_layer)
                                for dropout in [0]:
                                    parser.set_defaults(dropout=dropout)   
                                    for bs in [8,16,32]: # 8,16,32,64,128,256
                                        parser.set_defaults(batch_size=bs)
                                        for lr in [1e-3]: # 5e-4, 1e-4, 5e-5
                                            parser.set_defaults(lr=lr)
                                            args=[]
                                            args = parser.parse_args(args=args)  
                 
                                            train(args, None) 
