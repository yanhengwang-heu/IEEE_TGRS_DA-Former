import argparse
import torch
import torch.nn as nn
from scipy.io import loadmat
from vit_pytorch import Encoder_unshare,Encoder_share,Decoder_share,Decoder_unshare,Change_detection
from data_processing import SLset
import numpy as np
from torch.utils import data
from einops import rearrange
import matplotlib.pyplot as plt
import os
import torch.utils.data as Data
from sklearn.metrics import confusion_matrix
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

parser = argparse.ArgumentParser("HSI")
parser.add_argument('--dataset', choices=['BayArea', 'Barbara', 'river', 'farmland','Hermiston'], default='BayArea', help='dataset to use')
parser.add_argument('--flag_test', choices=['test', 'encoder1','encoder2','encoder2cd'], default='encoder2cd', help='testing mark')
parser.add_argument('--gpu_id', default='0', help='gpu id')
parser.add_argument('--seed', type=int, default=1, help='number of seed')
parser.add_argument('--batch_size', type=int, default=64, help='number of batch size')
parser.add_argument('--test_freq', type=int, default=5, help='number of evaluation')
parser.add_argument('--patch_size', type=int, default=9, help='patches_size')
parser.add_argument('--epoches', type=int, default=600, help='epoch number')
parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.95, help='gamma')
parser.add_argument('--ae_number', type=int, default=2000, help='autoencoder_number')
parser.add_argument('--cd_number', type=int, default=100, help='cd_number')
parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
parser.add_argument('--dim', type=int, default=64, help='patch encoder dimmention')
parser.add_argument('--dim_c', type=int, default=64, help='channel encoder dimmention')
parser.add_argument('--depth', type=int, default=2, help='patch encoder depth')
parser.add_argument('--depth_c', type=int, default=3, help='channel encoder depth')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
parser.add_argument('--heads', type=int, default=4, help='patch encoder heads')
parser.add_argument('--heads_c', type=int, default=8, help='channel encoder heads')
parser.add_argument('--dim_head', type=int, default=16, help='patch encoder dim_heads')
parser.add_argument('--dim_head_c', type=int, default=4, help='channel encoder dim_heads')

args = parser.parse_args()

#####加载数据#####
if args.dataset == 'BayArea':
    data_t1 = loadmat("./data/BayArea/Bay_Area_2013.mat")['HypeRvieW']
    data_t2 = loadmat("./data/BayArea/Bay_Area_2015.mat")['HypeRvieW']
    data_label = loadmat("./data/BayArea/bayArea_gtChanges2.mat")['HypeRvieW']
    uc_position = np.array(np.where(data_label==2)).transpose(1,0)
    c_position = np.array(np.where(data_label==1)).transpose(1,0)
    print((uc_position.shape[0],c_position.shape[0]))

elif args.dataset == 'Barbara':
    data_t1 = loadmat("./data/Barbara/barbara_2013.mat")['HypeRvieW']
    data_t2 = loadmat("./data/Barbara/barbara_2014.mat")['HypeRvieW']
    data_label = loadmat("./data/Barbara/barbara_gtChanges.mat")['HypeRvieW']
    uc_position = np.array(np.where(data_label==2)).transpose(1,0)
    c_position = np.array(np.where(data_label==1)).transpose(1,0)
    print((uc_position.shape[0],c_position.shape[0]))

elif args.dataset == 'river':
    data_t1 = loadmat('./data/river_dataset/river_before.mat')['river_before']
    data_t2 = loadmat('./data/river_dataset/river_after.mat')['river_after']
    data_label = loadmat('./data/river_dataset/groundtruth.mat')['lakelabel_v1']
    uc_position = np.array(np.where(data_label==0)).transpose(1,0)
    c_position = np.array(np.where(data_label==255)).transpose(1,0)
    print((uc_position.shape[0],c_position.shape[0]))
    data_label = (data_label-data_label.min())/(data_label.max()-data_label.min())
    data_label[data_label==0]=2
elif args.dataset == 'farmland':
    data_t1 = loadmat('./data/farmland/China_Change_Dataset.mat')['T1']
    data_t2 = loadmat('./data/farmland/China_Change_Dataset.mat')['T2']
    data_label = loadmat('./data/farmland/China_Change_Dataset.mat')['Binary']
    uc_position = np.array(np.where(data_label==0)).transpose(1,0)
    c_position = np.array(np.where(data_label==1)).transpose(1,0)
    print((uc_position.shape[0],c_position.shape[0]))
    data_label[data_label==0]=2
elif args.dataset == 'Hermiston':
    data_t1 = loadmat('../autoencoder/data/Hermiston/hermiston2004.mat')['HypeRvieW']
    data_t2 = loadmat('../autoencoder/data/Hermiston/hermiston2007.mat')['HypeRvieW']
    data_label = loadmat('../autoencoder/data/Hermiston/label_5classes.mat')['gt5clasesHermiston']
    data_t1 = np.concatenate((data_t1[:,:,:58],data_t1[:,:,76:]),2)
    data_t2 = np.concatenate((data_t2[:,:,:58],data_t2[:,:,76:]),2)
    uc_position = np.array(np.where(data_label==0)).transpose(1,0)
    c_position = np.array(np.where(data_label!=0)).transpose(1,0)
    print((uc_position.shape[0],c_position.shape[0]))
    data_label[data_label!=0]=1
    data_label[data_label==0]=2
else:
    raise ValueError("Unkknow dataset")


train_data = SLset(data_t1,data_t2,c_position,uc_position,args.patch_size,args.cd_number,args.ae_number)
train_data = Data.DataLoader(train_data,batch_size=args.batch_size, shuffle=True)


height,width,band = data_t1.shape
print("height={},width={},band={}".format(height,width,band))



#####加载模型#####
class Model(nn.Module):
    def __init__(self,band,patch_size,dim,depth,heads,dim_c,depth_c,heads_c,dim_head,dim_head_c,dropout):
        super().__init__()
        self.eu1 = Encoder_unshare(band,dim_c,dropout)
        self.eu2 = Encoder_unshare(band,dim_c,dropout)
        self.es = Encoder_share(patch_size,dim,depth,heads,dim_c,depth_c,heads_c,dim_head,dim_head_c,dropout)
        self.ds = Decoder_share(dim,dim_c,heads,dim_head,patch_size,dropout)
        self.du1 = Decoder_unshare(dim_c,band)
        self.du2 = Decoder_unshare(dim_c,band)
        self.cd = Change_detection(dim,dim_c,dropout)

    def forward(self,x1,x2):

        x1 = self.eu1(x1)
        c1,f1 = self.es(x1)
        d1 = self.ds(f1)
        out1 = self.du1(d1)

        
        d12 = self.du2(d1)
        d12 = self.eu2(d12)
        c12,d12 = self.es(d12)

        x2 = self.eu2(x2)
        c2,f2 = self.es(x2)
        d2 = self.ds(f2)
        out2 = self.du2(d2)

        d21 = self.du1(d2)
        d21 = self.eu1(d21)
        c21,d21 = self.es(d21)

        out = self.cd(f1,f2)
        return out1,out2,d12,d21,out,f1,f2

model = Model(band,args.patch_size,args.dim,args.depth,args.heads,args.dim_c,args.depth_c,args.heads_c,args.dim_head,args.dim_head_c,args.dropout).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epoches//20, gamma=args.gamma)

MSELoss = nn.MSELoss()
# criterion
criterion = nn.CrossEntropyLoss().cuda()
# optimizer
print("开始训练")

#####训练模型#####
for i in range(args.epoches):
    for j, (train_data1,train_data2,train_label) in  enumerate(train_data):
        train_data1 = train_data1.to(torch.float32).cuda()
        train_data2 = train_data2.to(torch.float32).cuda()
        train_label = train_label.long().cuda()
        
        out1,out2,d12,d21,out,f1,f2 = model(train_data1,train_data2)
        mask_position = torch.where(train_label!=-1)
        out = out[mask_position]
        train_label = train_label[mask_position]
        loss1=MSELoss(out1,train_data1)
        loss2=MSELoss(out2,train_data2)
        loss3=MSELoss(d12,f1)
        loss4=MSELoss(d21,f2)
        loss_cd=criterion(out,train_label)
        loss = loss1+loss2+0.4*loss_cd+loss3+loss4
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()
    print("Epoch: {:03d} ae1_loss: {:.4f} ae2_loss: {:.4f} cd_loss: {:.4f}"
                        .format(i+1, loss1, loss2, loss_cd))

def save_model(save_path, iteration, model):
    torch.save({'iteration': iteration,
                'model_dict': model.state_dict()},
                save_path)
    print("model save success")

save_model("logs/model_{}_{}_{}_{}.pth".format(args.dataset,args.patch_size,args.dim,args.dim_c), args.epoches, model)




