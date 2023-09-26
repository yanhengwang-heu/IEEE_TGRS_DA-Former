import torch.utils.data as data
import numpy as np
from einops import rearrange

class SLset(data.Dataset):
    def __init__(self,data1,data2,c_pos, uc_pos, size,nums,nums_ae):
        super(SLset,self).__init__()
        self.data1 = data1
        self.data2 = data2
        selected_c = np.random.choice(c_pos.shape[0], nums, replace = False)
        selected_uc = np.random.choice(uc_pos.shape[0], nums, replace = False)

        selected_uc_position = uc_pos[selected_uc]
        selected_c_position = c_pos[selected_c]

        self.img_cut1_uc=[]
        self.img_cut2_uc=[]
        self.img_cut1_c=[]
        self.img_cut2_c=[]
        self.img_cut1_ae=[]
        self.img_cut2_ae=[]


        h,w,c = self.data1.shape
        # print("height={0},width={1},band={2}".format(h, w, c))
        h1_sample = np.random.choice(h,nums_ae)
        w1_sample = np.random.choice(w,nums_ae)
        h2_sample = np.random.choice(h,nums_ae)
        w2_sample = np.random.choice(w,nums_ae)


        self.data1 = (self.data1-self.data1.min())/(self.data1.max()-self.data1.min())
        self.data2 = (self.data2-self.data2.min())/(self.data2.max()-self.data2.min())
        self.data1 = self.data1.transpose(2,0,1)
        self.data2 = self.data2.transpose(2,0,1)

        self.data1 = np.pad(self.data1,((0,0),(size//2,size//2),(size//2,size//2)),"edge")
        self.data2 = np.pad(self.data2,((0,0),(size//2,size//2),(size//2,size//2)),"edge")

        band,height,width = self.data1.shape
        print("pad_after: height={0},width={1},band={2}".format(height, width, band))

        for i in range(nums):
            img_cut1_uc = self.data1[:,selected_uc_position[i,0]:selected_uc_position[i,0]+size,selected_uc_position[i,1]:selected_uc_position[i,1]+size]
            img_cut1_c = self.data1[:,selected_c_position[i,0]:selected_c_position[i,0]+size,selected_c_position[i,1]:selected_c_position[i,1]+size]
            img_cut2_uc = self.data2[:,selected_uc_position[i,0]:selected_uc_position[i,0]+size,selected_uc_position[i,1]:selected_uc_position[i,1]+size]
            img_cut2_c = self.data2[:,selected_c_position[i,0]:selected_c_position[i,0]+size,selected_c_position[i,1]:selected_c_position[i,1]+size]

            img_cut1_uc = rearrange(img_cut1_uc, 'c h w -> (h w) c')
            img_cut1_c = rearrange(img_cut1_c, 'c h w -> (h w) c')
            img_cut2_uc = rearrange(img_cut2_uc, 'c h w -> (h w) c')
            img_cut2_c = rearrange(img_cut2_c, 'c h w -> (h w) c')


            self.img_cut1_uc.append(img_cut1_uc)
            self.img_cut1_c.append(img_cut1_c)
            self.img_cut2_uc.append(img_cut2_uc)
            self.img_cut2_c.append(img_cut2_c)
        for i in range(nums_ae):
            img_cut1 = self.data1[:,h1_sample[i]:h1_sample[i]+size,w1_sample[i]:w1_sample[i]+size]
            img_cut2 = self.data2[:,h2_sample[i]:h2_sample[i]+size,w2_sample[i]:w2_sample[i]+size]

            img_cut1 = rearrange(img_cut1, 'c h w -> (h w) c')
            img_cut2 = rearrange(img_cut2, 'c h w -> (h w) c')

            self.img_cut1_ae.append(img_cut1)
            self.img_cut2_ae.append(img_cut2)

        self.label = list(np.hstack((-np.ones(nums_ae),np.zeros(nums),np.ones(nums))))
        self.img1 = self.img_cut1_ae+self.img_cut1_uc+self.img_cut1_c
        self.img2 = self.img_cut2_ae+self.img_cut2_uc+self.img_cut2_c

    def __getitem__(self,index):
        return self.img1[index], self.img2[index], self.label[index]
    
    def __len__(self):
        return len(self.img1)            




