
import argparse
import os
import torch
import torchvision.transforms.v2 as v2
from pathlib import Path
import os

from dlvc.models.class_model import DeepClassifier 
from dlvc.metrics import Accuracy
from dlvc.trainer import ImgClassificationTrainer
from dlvc.datasets.cifar10 import CIFAR10Dataset
from dlvc.datasets.dataset import Subset
import matplotlib.pyplot as plt
import tarfile
from torchvision.models import resnet18
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
import numpy as np
import copy
import torchvision.transforms.functional as F
from dlvc.models.vit import ViT



def train(args):

    train_transform = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Normalize(mean = [0.485, 0.456,0.406], std = [0.229, 0.224, 0.225])])
    
    val_transform = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Normalize(mean = [0.485, 0.456,0.406], std = [0.229, 0.224, 0.225])])
    
    
    train_data = CIFAR10Dataset('dlvc/datasets/cifar-10-python.tar.gz', 'training',transform=train_transform)
    val_data = CIFAR10Dataset('dlvc/datasets/cifar-10-python.tar.gz', 'validation',transform=val_transform)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = DeepClassifier(ViT(
        img_size=32,
        patch_size=4,
        num_classes=10,
        dim=256,
        depth=6,
        heads=8,
        mlp_dim=512,
        dropout=0.1,
        emb_dropout=0.1
    ))
    model.to(device)
    
    optimizer = SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    loss_fn = torch.nn.CrossEntropyLoss()
    
    train_metric = Accuracy(classes=train_data.classes)
    val_metric = Accuracy(classes=val_data.classes)

    val_frequency = 5

    model_save_dir = Path("saved_models")
    model_save_dir.mkdir(exist_ok=True)

    trainer = ImgClassificationTrainer(model, 
                    optimizer,
                    loss_fn,
                    lr_scheduler,
                    train_metric,
                    val_metric,
                    train_data,
                    val_data,
                    device,
                    args.num_epochs, 
                    model_save_dir,
                    batch_size=128, 
                    val_frequency = val_frequency)
    trainer.train()
    trainer.save_accuracy_history()

    trainer.plot_accuracy_history(trainer.train_accuracy_history, trainer.val_accuracy_history)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('-d', '--gpu_id', default='0', type=str, help='index of which GPU to use')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimization')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay for regularization')
    parser.add_argument('--lr_step_size', type=int, default=5, help='Step size for learning rate scheduler')
    parser.add_argument('--lr_gamma', type=float, default=0.1, help='Gamma value for learning rate scheduler')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of epochs')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    args.gpu_id = 0                                                 #defautl
    args.num_epochs = 30                                            # 30
    ##
    args.batch_size =  128                                            #  128
    args.learning_rate =  0.09                                    #0.001 #0.03
    args.momentum =    0.9                                            #0.9
    args.weight_decay =    0.0001                                         #0.0001
    args.lr_step_size =    5                                          #5
    args.lr_gamma =     0.1                                             #0.1
    ##

    train(args)