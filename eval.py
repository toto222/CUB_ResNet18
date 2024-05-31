import torch
import torch.nn as nn
from dataset import dataset
from torch.utils.data import DataLoader
import torchvision.models as models
from tqdm import tqdm
import argparse

def parse_option():
    parser = argparse.ArgumentParser('Vision Models for Classification')
    # Training setting
    parser.add_argument('--batch_size', type=int, default=64, 
                    help='batch_size')
    # dataset & model
    parser.add_argument('--file', type=str, default='./save/best.pth',
                    help='model pth path')
    parser.add_argument('--root', type=str, default='./dataset/CUB_200_2011',
                help='dataset path')

    args = parser.parse_args()

    return args

def main():

    args = parse_option()

    batch_size = args.batch_size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    val_data = dataset(path=args.root,split='val')
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=8)

    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(in_features=512, out_features=200, bias=True)
    model.to(device)
    model.load_state_dict(torch.load(args.file))

    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(val_loader, total=len(val_loader)):
            images, y = data
            images, y = images.to(device), y.to(device)
            outputs = model(images)

            predicted = torch.argmax(outputs, 1)
            labels = torch.argmax(y, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total

    print(f'acc: {accuracy}')

if __name__=="__main__":
    main()