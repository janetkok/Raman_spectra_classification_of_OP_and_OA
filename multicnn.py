import os
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from model.resnet import Multi_ResNet
import copy
import numpy as np
import pandas as pd
from tqdm import trange


from parser import parse_args_train
from data.dataset import RamanMultiChannelDataset

from datetime import datetime
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
from utils import *
from sklearn.metrics import precision_recall_fscore_support,accuracy_score


      
def train(model, data_loader, criterion, optimizer, num_epochs=100,saver=None,best_epoch_val_error=None,foldMain='',phases = ["train","val"],filename="cv.csv"):
  train_metrics = dict()
  eval_metrics = dict()
  best_metric = None
  best_epoch = None
  best_val_error = None
  early_test_break=False

  writer = SummaryWriter(log_dir=os.path.join(foldMain, 'logs'))

  for epoch in trange(num_epochs, desc="Epochs"):
    result = []
    for phase in phases:
      outputs = []
      targets = []
      if phase == "train" or phase =="trainval":     # training mode
        model.train()
      else:     #  validation mode
        model.eval()

      # keep track of training and validation loss
      running_loss = 0.0

      for data, target in data_loader[phase]:
        # load the data and target to respective device
        data, target = data.cuda(0), target.cuda(0)

        with torch.set_grad_enabled(phase == "train" or phase =="trainval"):

            # feed the input
            output = model(data.float())
 
            # calculate the loss
            loss = criterion(output, target)

            if phase == "train" or phase =="trainval":
                # zero the grad to stop it from accumulating
                optimizer.zero_grad()
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # update the model parameters
                optimizer.step()

            topk = output.topk(1)[1]
            outputs=np.append(outputs,topk.cpu().numpy())
            targets=np.append(targets,target.cpu().numpy())
                
        running_loss += loss.item() * data.size(0)

      prec,rec,f1,_ = precision_recall_fscore_support(targets,outputs,average='binary')
      acc = accuracy_score(targets, outputs)
      epoch_loss = running_loss / len(data_loader[phase].dataset)
      lrl = [param_group['lr'] for param_group in optimizer.param_groups]
      lr = sum(lrl) / len(lrl)

      log_stats(writer,epoch,phase, epoch_loss, acc)


      result.append('{} LR: {:.4f} Loss: {:.4f} Acc: {:.4f} Acc1: {:.4f}'.format(phase, lr,epoch_loss, acc, acc))
      if phase=="train" or phase =="trainval":
          train_metrics = OrderedDict([('loss', epoch_loss), ('acc', acc), ('acc1', acc), ('prec', prec), ('recall', rec),('f1', f1)])
      else:
          eval_metrics = OrderedDict([('loss', epoch_loss), ('acc', acc), ('acc1', acc),('prec', prec), ('recall', rec),('f1', f1)])

          if phase=="val" and (best_val_error is None or best_val_error>=epoch_loss):
              best_epoch_val_error = epoch
              best_val_error = epoch_loss
              print('*** Best eval loss: {0} (epoch {1})'.format(best_val_error, best_epoch_val_error))

          if saver is not None and epoch==best_epoch_val_error: #during test time only
              # save checkpoint with eval metric
              save_metric = acc
              best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)  
              print('*** Best metric based on val error: {0} (epoch {1})'.format(best_metric, best_epoch))
              early_test_break = True
              
    log_lr(writer,optimizer,epoch)
    update_summary(epoch, train_metrics, eval_metrics, filename=os.path.join(foldMain, filename), write_header=(epoch==0))
    print(result)

    if early_test_break==True:
      break
  
  return eval_metrics,best_epoch_val_error


def main():
    #initialisation 
    args, args_text = parse_args_train()
    cv_metrics = dict(val_loss=[],val_f1=[],test_loss=[],test_f1=[])
    ds = pd.read_csv(args.dataset)
    feature_names = list(ds.columns[2:])
    folder_name = args.dataset.split("/")[-1].split(".csv")[0] 
    foldMain = './output/multicnn/'+folder_name

    #seed
    os.environ['PYTHONHASHSEED']=str(args.seed)
    seed_worker(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)  

    # to split by patient id
    ids = ds['id'].unique()
    n_classes = len(ds['label'].unique())
    rows_list = []
    for i in range(len(ids)):
      rows_list.append(ds.loc[ds['id'].isin([i])].iloc[0,:])
    firstR = pd.DataFrame(rows_list)
    firstR.columns = ds.columns

    splitter = StratifiedKFold3(n_splits=6,shuffle=True,random_state=args.seed)
    for fold, (train_id, val_id, test_id) in enumerate(splitter.split(firstR.iloc[:,2:], firstR['label'])):
        
        # assign patient's data to different partition according to their ids
        train_id=train_id.tolist()
        train_idx=ds.loc[ds['id'].isin(train_id)].index.tolist()
        val_id=val_id.tolist()
        val_idx=ds.loc[ds['id'].isin(val_id)].index.tolist()
        trainval_id = train_id + val_id
        trainval_idx=ds.loc[ds['id'].isin(trainval_id)].index.tolist()
        test_id=test_id.tolist()
        test_idx=ds.loc[ds['id'].isin(test_id)].index.tolist()
        
        print('fold '+ str(fold))

        exp_name = '-'.join([str(fold),
                        datetime.now().strftime("%Y%m%d-%H%M%S")])
        output_dir = get_outdir(foldMain, exp_name)
        print(output_dir)

        ds_dropId = ds.drop(['id'], axis=1)

        # CNN parameters
        input_dim = ds_dropId.shape[1]-1

        model = Multi_ResNet(input_dim=input_dim, n_classes=n_classes,channel=len(args.channel)+1).cuda()
        print(model)
        modelVal = copy.deepcopy(model) # copy model for cross validation use
        # Set up Adam optimizer
        optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.5, 0.999))
        optimizerVal = optim.Adam(modelVal.parameters(), lr=1e-4, betas=(0.5, 0.999))
        # loss
        criterion = nn.CrossEntropyLoss().cuda()

        trainset = RamanMultiChannelDataset(ds_dropId, train_idx,channel=args.channel)
        valset = RamanMultiChannelDataset(ds_dropId, val_idx,channel=args.channel)
        trainvalset = RamanMultiChannelDataset(ds_dropId, trainval_idx,channel=args.channel)
        testset = RamanMultiChannelDataset(ds_dropId, test_idx,channel=args.channel)
        print(f"trainset len {len(trainset)} valset len {len(valset)} trainval len {len(trainvalset)} testset len {len(testset)}")

        #seed
        seed_worker(args.seed)

        kwargs = {'num_workers': args.threads, 'pin_memory': True,'worker_init_fn':seed_worker, 'generator':g}
        dataloader = {"train": DataLoader(trainset, shuffle=True, batch_size=args.train_batch,**kwargs),
                  "val": DataLoader(valset, shuffle=False, batch_size= args.valid_batch,**kwargs),
                  "trainval": DataLoader(trainvalset, shuffle=True, batch_size=args.train_batch,**kwargs),
                  "test": DataLoader(testset, shuffle=False, batch_size= args.valid_batch,**kwargs)}
        print(f"trainloader len {len(dataloader['train'])} valloader len {len(dataloader['val'])} trainvalloader len {len(dataloader['trainval'])} testloader len {len(dataloader['test'])}")
      
        # save training config
        saver = CheckpointSaver(model=model, optimizer=optimizer, args=args,
            checkpoint_dir=output_dir,  decreasing=False, max_history=args.checkpoint_hist)
        
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)

        # cross validate(trainset and valset) to pick optimal epoch
        eval_metrics,best_epoch_val_error  = train(modelVal,dataloader, criterion, optimizerVal, num_epochs=args.epochs,foldMain=foldMain,phases=["train","val"],filename="cv.csv")
        # cross validate(trainset combined with valset and testset) using the optimal epoch
        et_metrics,_ = train(model,dataloader, criterion, optimizer, num_epochs=args.epochs,saver=saver,best_epoch_val_error =best_epoch_val_error,foldMain=foldMain,phases=["trainval","test"],filename="ct.csv")
        cv_metrics['val_loss'].append(eval_metrics['loss'])
        cv_metrics['val_f1'].append(eval_metrics['acc'])
        cv_metrics['test_loss'].append(et_metrics['loss'])
        cv_metrics['test_f1'].append(et_metrics['acc'])

        # Integrated gradients
        model = Multi_ResNet(input_dim=input_dim, n_classes=n_classes,channel=len(args.channel)+1).cuda()
        # load best model selected based on val error
        print("=> loading checkpoint '{}'".format(output_dir+'/model_best.pth.tar'))
        checkpoint = torch.load(output_dir+'/model_best.pth.tar')
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(output_dir+'/model_best.pth.tar', checkpoint['epoch']))
        model = nn.DataParallel(model) #multigpu
        ig(model,dataloader,feature_names,output_dir,args.seed)
      
    # cross validation results
    avg_metrics = OrderedDict([('val_loss', np.mean(cv_metrics['val_loss'])),('val_f1', np.mean(cv_metrics['val_f1']))])
    print(avg_metrics)
    update_cv('avg_metrics',avg_metrics,  os.path.join(foldMain, 'cv.csv'), write_header=True)

    avg_metrics = OrderedDict([('test_loss', np.mean(cv_metrics['test_loss'])),('test_f1', np.mean(cv_metrics['test_f1']))])
    print(avg_metrics)
    update_cv('avg_metrics',avg_metrics,  os.path.join(foldMain, 'ct.csv'), write_header=True)
    
    std_metrics =  OrderedDict([('sd val loss', np.std(cv_metrics['val_loss'])), ('sd val acc', np.std(cv_metrics['val_f1']))])
    print(std_metrics)
    update_cv('std_metrics',std_metrics, os.path.join(foldMain, 'cv.csv'), write_header=True)

    std_metrics =  OrderedDict([('sd test loss', np.std(cv_metrics['test_loss'])), ('sd test acc', np.std(cv_metrics['test_f1']))])
    print(std_metrics)
    update_cv('std_metrics',std_metrics, os.path.join(foldMain, 'ct.csv'), write_header=True)


    #summarising feature importances across all folds and save it in fold 0
    feat = pd.DataFrame()
    for i,fold in enumerate(os.listdir(foldMain)):
      if fold[0].isdigit():
        df=pd.read_csv(foldMain+'/'+fold+'/feat.csv')
        df = df.sort_values(by="feature",ascending=True)
        if fold.startswith('0'):
            feat.insert(0, 'feature', df['feature']) #insert column names
        feat = pd.concat([feat,df['imp_total']],axis=1)

    sum_column = feat.iloc[:,1] + feat.iloc[:,2] + feat.iloc[:,3] + feat.iloc[:,4] + feat.iloc[:,5] 
    feat["total"] = sum_column

    feat = feat.sort_values(by="total",ascending=False)
    feat.to_csv(foldMain+'/cvFeat.csv', index=False)
                     
if __name__ == '__main__':
    main()