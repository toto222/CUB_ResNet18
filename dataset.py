import os
from PIL import Image
import torch
from torchvision import transforms

# def read_dataset(path):
transform = transforms.Compose([
    transforms.Resize(512),
    # transforms.RandomResizedCrop(224),
    # transforms.RandomHorizontalFlip(), 
    transforms.RandomCrop(448),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])
        
class dataset:
    def __init__(self, path='./dataset/CUB_200_2011', split='train'):
        self.data=[]
        self.path = path
        self.split=split
        self.dict={}
        self.read_dataset()
        self.prepare_dic()
              
    def read_dataset(self):
        # split = os.path.join(self.path,'train_test_split.txt')
        split = os.path.join(self.path,'data_split.txt')
        
        assert os.path.exists(split) ,'Wrong file path'
        assert self.split in ['train','val'], "split should be \'train' or \'val'"

        data=[]

        tag = '1' if self.split=='train' else '0'
        with open(split,'r') as file:
            for line in file.readlines():
                idx, tp = line[:-1].split(' ')
                if tp==tag:
                    data.append(int(idx)) 
                
        self.data=data
        
    def prepare_dic(self):
        images = os.path.join(self.path,'images.txt')
        
        assert os.path.exists(images), 'Wrong file path'
        dic={}
        with open(images,'r') as file:
            lines = file.readlines()
            for n in self.data:
                line = lines[n-1][:-1]
                name = line.split(' ')[-1]
                dic[n]=name
        self.dict = dic
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        n = self.data[idx]
        name = os.path.join(self.path, 'images', self.dict[n])
        lable = int(self.dict[n].split('.')[0])-1
        y = torch.zeros(200)
        y[lable]=1
        pic = Image.open(name).convert('RGB')

        return transform(pic), y
        

        