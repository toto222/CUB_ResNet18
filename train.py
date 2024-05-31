import torch
import torch.nn as nn
import torchvision.models as models
from dataset import dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse

def parse_option():
    parser = argparse.ArgumentParser('Vision Models for Classification')
    # Training setting
    parser.add_argument('--batch_size', type=int, default=128, 
                    help='batch_size')
    parser.add_argument('--epoch', type=int, default=20,
                    help='number of training epochs')
    parser.add_argument('--lr', type=float, default=5e-4,
                    help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.)
    # dataset & model
    parser.add_argument('--root', type=str, default='./dataset/CUB_200_2011',
                    help='dataset path')
    parser.add_argument('--seed', type=int, default=850011,
                    help='seed for initializing training')
    parser.add_argument('--pretrained',default=False, action="store_true")
    parser.add_argument('--save_dir', type=str, default='./save',
                    help='path to save models')

    args = parser.parse_args()

    return args


# import 
def main():

    args = parse_option()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    writer = SummaryWriter('./train_logs_'+ datetime.now().strftime("%m%d-%H%M"))
    
    batch_size = args.batch_size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_data = dataset(path=args.root, split='train')
    val_data = dataset(path=args.root, split='val')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_data, batch_size=batch_size//2, shuffle=False, num_workers=8)
    
    if args.pretrained:
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(in_features=512, out_features=200, bias=True)
        params_fc = [p for p in model.fc.parameters()] 
        params_others = [p for p in model.parameters() if id(p) not in [id(p) for p in params_fc]]
        optimizer = torch.optim.Adam([
                {'params': params_fc, 'lr': args.lr},
                {'params': params_others, 'lr': args.lr*0.1},
            ], weight_decay=args.weight_decay,)
    else:
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(in_features=512, out_features=200, bias=True)
        optimizer = torch.optim.Adam(model.parameters(),lr=5e-4,weight_decay=args.weight_decay)
    
    

    model.to(device)
    epoch = args.epoch
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=1e-8)
    criterion = nn.CrossEntropyLoss().to(device)

    
    best_acc = 0
    for ep in range(1,epoch+1):
        print(f'Training for epoch {ep}')
        # import pdb;pdb.set_trace()
        writer.add_scalar('lr',scheduler.get_last_lr()[0],ep)
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
        # scheduler.step()
        
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

        if accuracy > best_acc:
            torch.save(model.state_dict(), './save/best.pth')
            best_acc = accuracy
    
    torch.save(model.state_dict(), './save/last.pth')
    print(f'best acc: {best_acc}')
        
if __name__=="__main__":
    main()    

    