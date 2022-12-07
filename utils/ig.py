from captum.attr import IntegratedGradients
from matplotlib import pyplot as plt
import numpy as np
import torch
import pandas as pd
from .seed import seed_worker




def ig(model,data_loader,feature_names,output_dir,seed):
  x=[]
  for data, target in data_loader['test']:
    x.append(data.squeeze().numpy())
  
  x=np.array(x).astype(np.float64)
  test_input_tensor = torch.from_numpy(x).type(torch.FloatTensor).cuda()
  ig = IntegratedGradients(model.cuda())
  test_input_tensor.requires_grad_()
  importances=[]
  for target in [0,1]:
    attr, _ = ig.attribute(test_input_tensor,target=target, return_convergence_delta=True,internal_batch_size=1)
    attr = attr.detach().cpu().numpy()
    importance = np.mean(attr, axis=0)
    importance = np.sum(abs(importance),axis=0)
    importances.append(importance)

  data = {'feature':feature_names,'importance0':importances[0],'importance1':importances[1]}

  df=pd.DataFrame(data)
  sum_column = df["importance0"] + df["importance1"] #sum up importances looking at target 0 and target 1
  df["imp_total"] = sum_column
  
  temp = df.sort_values(by="imp_total",ascending=False)
  temp.to_csv(output_dir+'/feat.csv', index=False)

