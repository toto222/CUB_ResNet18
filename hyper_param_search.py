import torch
import torch.nn as nn
import torchvision.models as models
from dataset import dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


# import 
def main():
    
    batch_size = 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_data = dataset()
    val_data = dataset(split='val')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_data, batch_size=batch_size//2, shuffle=False, num_workers=8)
    
    # model = models.alexnet(pretrained=True)
    # num_ftrs = model.classifier[6].in_features
    # model.classifier[6] = nn.Linear(num_ftrs, 200)
    # params_fc = [model.classifier[6]]
    # params_others = [p for p in model.parameters() if id(p) not in [id(p) for p in params_fc]]


    
    # lr = [0.1,0.05,0.01,] # for cuda:0
    # lr = [0.005,0.001,0.0005] # for cuda:1
    lr = [0.1,0.05,0.01,]+[0.005,0.001,0.0005] 
    times = [0.5, 0.1, 0.05, 0.01,0]
    
    for lr_fc in lr:
        for time in times:
            lr_others = lr_fc * time

            model = models.resnet18(pretrained=True)
            model.fc = nn.Linear(in_features=512, out_features=200, bias=True)
            params_fc = [p for p in model.fc.parameters()] 
            params_others = [p for p in model.parameters() if id(p) not in [id(p) for p in params_fc]]

            model.to(device)
    
            writer = SummaryWriter(f'./param_serach/logs_{lr_fc:.1e}_{lr_others:.1e}')
            optimizer = torch.optim.Adam([
                    {'params': params_fc, 'lr': lr_fc},
                    {'params': params_others, 'lr': lr_others},
                ], weight_decay=0,)

            # optimizer = torch.optim.SGD([
            #     {'params': params_fc, 'lr': lr_fc},
            #     {'params': params_others, 'lr': lr_others},
            # ], weight_decay=0.0005,momentum=0.9)
            
            criterion = nn.CrossEntropyLoss()
            epoch = 50
            
            best_acc = 0
            for ep in range(1,epoch+1):
                print(f'Training for epoch {ep}')
                running_loss = 0.
                for batch in tqdm(train_loader,total=len(train_loader)):
                    x,y=batch
                    x = x.to(device)
                    y = y.to(device)

                    optimizer.zero_grad()
                    y_pre = model(x)
                    loss = criterion(y_pre, y)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                print(f'Train loss: {running_loss}')
                
                print(f'Validating for epoch {ep}')
                correct = 0
                total = 0
                val_loss = 0
                with torch.no_grad():
                    for data in tqdm(val_loader, total=len(val_loader)):
                        images, y = data
                        images, y = images.to(device), y.to(device)
                        outputs = model(images)
                        val_loss += criterion(outputs, y).item()

                        predicted = torch.argmax(outputs, 1)
                        labels = torch.argmax(y, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                    accuracy = 100 * correct / total
                print(f'Val loss: {val_loss}')
                print(f'Val Accuracy:{accuracy}')
                
                writer.add_scalar('Loss/train',running_loss, ep)
                writer.add_scalar('Loss/validation',val_loss, ep)
                writer.add_scalar('Accuracy', accuracy, ep)
                if accuracy>best_acc:
                    best_acc = accuracy
                    
            with open('parmp_search.txt','a') as file:
                file.write(f'best acc of {lr_fc:.1e} with {lr_others:.1e}: {best_acc}\n')

            writer.close()
        
if __name__=="__main__":
    main()    

    