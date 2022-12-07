import os
import torch


import numpy as np
from sklearn import tree,svm
from sklearn.preprocessing import MinMaxScaler

import pandas as pd



from parser import parse_args_train

from datetime import datetime
from collections import OrderedDict

from utils import *
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def formatNumber(n): return n if n % 1 else int(n)


def main():
    #initialisation 
    args, args_text = parse_args_train()
    acc = prec = rec = f1 = 0
    cv_metrics = dict(acc=[], prec=[],rec=[],f1=[])
    ds = pd.read_csv(args.dataset)
    folder_name = args.dataset.split("/")[-1].split(".csv")[0] 
    foldMulticnn = './output/multicnn/'+folder_name
    foldMain = './output/'+args.method+'_nf'+str(args.num_feat)+'/'+folder_name

    # seed
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    seed_worker(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)

    # extract top n features
    feat = pd.read_csv(foldMulticnn+'/cvFeat.csv')['feature'].tolist()
    topFeat = [float(x) for x in feat[0:args.num_feat]]
    sortTop = sorted(topFeat, reverse=True)
    sortTop = [str(formatNumber(x)) for x in sortTop]
    ds = ds[['id', 'label']+sortTop]
    
    # to split by patient id
    ids = ds['id'].unique()
    rows_list = []
    for i in range(len(ids)):
        rows_list.append(ds.loc[ds['id'].isin([i])].iloc[0, :])
    firstR = pd.DataFrame(rows_list)
    firstR.columns = ds.columns
    
    splitter = StratifiedKFold3(n_splits=6,shuffle=True,random_state=args.seed)
    for fold, (train_id, val_id, test_id) in enumerate(splitter.split(firstR.iloc[:,2:], firstR['label'])):
        
        # assign patient's data to different partition according to their ids
        train_id=train_id.tolist()
        val_id=val_id.tolist()
        trainval_id = train_id + val_id
        trainval_idx=ds.loc[ds['id'].isin(trainval_id)].index.tolist()
        test_id=test_id.tolist()
        test_idx=ds.loc[ds['id'].isin(test_id)].index.tolist()

        print('fold '+ str(fold))

        exp_name = '-'.join([
            str(fold),
            datetime.now().strftime("%Y%m%d-%H%M%S")
        ])
        output_dir = get_outdir(foldMain, exp_name)
        print(output_dir)

        ds_dropId = ds.drop(['id'], axis=1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        ds_dropId.iloc[:, 1:] = scaler.fit_transform(ds_dropId.iloc[:, 1:])
        train_x = ds_dropId.iloc[trainval_idx, 1:]
        train_y = ds_dropId.iloc[trainval_idx, 0]
        test_x = ds_dropId.iloc[test_idx, 1:]
        test_y = ds_dropId.iloc[test_idx, 0]

        clf = svm.SVC()
        if args.method=="dt":
            clf = tree.DecisionTreeClassifier()
        
        clf.fit(train_x, train_y)
        predTest_y = clf.predict(test_x)

        acc = accuracy_score(predTest_y, test_y)
        prec, rec, f1, _ = precision_recall_fscore_support(
            test_y, predTest_y, average='binary')

        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)

        avg_metrics = OrderedDict(
            [('prec', prec), ('rec', rec), ('f1', f1), ('acc', acc)])
        print(avg_metrics)
        update_cv('avg_metrics', avg_metrics,  os.path.join(
            foldMain, 'summary.csv'), write_header=(fold == 0))
        cv_metrics['acc'].append(acc)
        cv_metrics['prec'].append(prec)
        cv_metrics['rec'].append(rec)
        cv_metrics['f1'].append(f1)

    # cross validation results
    avg_metrics = OrderedDict([('acc', np.mean(cv_metrics['acc'])),('prec', np.mean(cv_metrics['prec'])),('rec', np.mean(cv_metrics['rec'])),('f1', np.mean(cv_metrics['f1']))])
    print(avg_metrics)
    update_cv('avg_metrics', avg_metrics,  os.path.join(
        foldMain, 'summary.csv'), write_header=True)

    std_metrics = OrderedDict([('acc', np.std(cv_metrics['acc'])),('prec', np.std(cv_metrics['prec'])),('rec', np.std(cv_metrics['rec'])),('f1', np.std(cv_metrics['f1']))])
    print(std_metrics)
    update_cv('std_metrics', std_metrics, os.path.join(
        foldMain, 'summary.csv'), write_header=True)


if __name__ == '__main__':
    main()
