# ----------------------------------------------------------------------------------------------------
# Reference https://github.com/uoguelph-mlrg/Cutout
# ----------------------------------------------------------------------------------------------------

# run train.py --dataset cifar10 --model resnet18 --data_augmentation --cutout --length 16
# run train.py --dataset cifar100 --model resnet18 --data_augmentation --cutout --length 8
# run train.py --dataset svhn --model wideresnet --learning_rate 0.01 --epochs 160 --cutout --length 20

import pdb
import argparse
import numpy as np
from tqdm import tqdm
import pickle
from sklearn.model_selection import train_test_split
import os
from PIL import Image

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR

from torchvision.utils import make_grid
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset

from util.misc import CSVLogger
from util.cutout import Cutout

from model.resnet import ResNet18
from model.wide_resnet import WideResNet

model_options = ['resnet18', 'wideresnet']
dataset_options = ['cifar10', 'cifar100', 'svhn']


RESULTS_DIR = './results'
if not os.path.exists(RESULTS_DIR):
    os.mkdir(RESULTS_DIR)

# --------------------------------------------------------------------------------------------
# Process Input Arguments
# --------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--dataset', '-d', default='cifar10',
                    choices=dataset_options)
parser.add_argument('--model', '-a', default='resnet18',
                    choices=model_options)
parser.add_argument('--batch_size', type=int, default=128,
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train (default: 20)')
parser.add_argument('--learning_rate', type=float, default=0.1,
                    help='learning rate')
parser.add_argument('--data_augmentation', action='store_true', default=False,
                    help='augment data by flipping and cropping')
parser.add_argument('--cutout', action='store_true', default=False,
                    help='apply cutout')
parser.add_argument('--n_holes', type=int, default=1,
                    help='number of holes to cut out from image')
parser.add_argument('--length', type=int, default=16,
                    help='length of the holes')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed (default: 1)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
cudnn.benchmark = True  # Should make training should go faster for large models

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

test_id = args.dataset + '_' + args.model

# print the input arguments:
for k, v in args.__dict__.items():
    print("{}: {}".format(k, v))


# ---------------------------------------------------------------------------------------
# Custom Dataset Loading
# ---------------------------------------------------------------------------------------
class Stat946TrainDataSet(Dataset):

    def __init__(self, x_data, y_labels, preprocessing=None):
        self.data = x_data
        self.labels = y_labels
        self.count = x_data.shape[0]
        self.preprocessing = preprocessing

    def __getitem__(self, index):

        img = self.data[index, ]
        img = Image.fromarray(img)

        if self.preprocessing is not None:
            img = self.preprocessing(img)

        return img, self.labels[index]

    def __len__(self):
        return self.count  # of how many examples(images?) you have


class Stat946TestDataSet(Dataset):
    """ No Labels """

    def __init__(self, x_data, preprocessing=None):
        self.data = x_data
        self.count = x_data.shape[0]
        self.preprocessing = preprocessing

    def __getitem__(self, index):

        img = self.data[index, ]
        img = Image.fromarray(img)

        if self.preprocessing is not None:
            img = self.preprocessing(img)

        return img

    def __len__(self):
        return self.count  # of how many examples(images?) you have


def get_stat_946_datasets(validation_data_split, train_preprocessing, test_preprocessing):

    base_dir = "./data/stat_946_data"

    if not os.path.exists(base_dir):
        raise Exception("Cannot find STAT 946 Data Files. Download files and place @ {}".format(base_dir))

    with open(os.path.join(base_dir, 'train_data'), 'rb') as f:
        train_data = pickle.load(f, encoding='bytes')
        train_label = pickle.load(f, encoding='bytes')

    with open(os.path.join(base_dir, 'test_data'), 'rb') as f:
        test_data = pickle.load(f, encoding='bytes')

    # Put data in correct format
    train_data = train_data.reshape((train_data.shape[0], 3, 32, 32))
    train_data = train_data.transpose(0, 2, 3, 1)
    x_train, x_validation, y_train, y_validation = \
        train_test_split(train_data, train_label, test_size=validation_data_split, random_state=args.seed)

    test_data = test_data.reshape((test_data.shape[0], 3, 32, 32))
    test_data = test_data.transpose(0, 2, 3, 1)
    x_test = test_data

    # Now Create PyTorch Data Sets
    train_ds = Stat946TrainDataSet(x_train, y_train, train_preprocessing)
    validation_ds = Stat946TrainDataSet(x_validation, y_validation, test_preprocessing)
    test_ds = Stat946TrainDataSet(x_test, test_preprocessing)

    return train_ds, validation_ds, test_ds


# ---------------------------------------------------------------------------------------
# Image Pre-processing
# ---------------------------------------------------------------------------------------
if args.dataset == 'svhn':
    normalize = transforms.Normalize(
        mean=[x / 255.0 for x in[109.9, 109.7, 113.8]],
        std=[x / 255.0 for x in [50.1, 50.6, 50.8]]
    )
else:
    normalize = transforms.Normalize(
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
    )

train_transform = transforms.Compose([])

if args.data_augmentation:
    train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
    train_transform.transforms.append(transforms.RandomHorizontalFlip())
train_transform.transforms.append(transforms.ToTensor())
train_transform.transforms.append(normalize)

if args.cutout:
    train_transform.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))

test_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize])

# ---------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------
if args.dataset == 'cifar10':
    num_classes = 10
    train_dataset = datasets.CIFAR10(root='data/',
                                     train=True,
                                     transform=train_transform,
                                     download=True)

    test_dataset = datasets.CIFAR10(root='data/',
                                    train=False,
                                    transform=test_transform,
                                    download=True)
elif args.dataset == 'cifar100':
    num_classes = 100
    # train_dataset = datasets.CIFAR100(root='data/',
    #                                   train=True,
    #                                   transform=train_transform,
    #                                   download=True)
    #
    # test_dataset = datasets.CIFAR100(root='data/',
    #                                  train=False,
    #                                  transform=test_transform,
    #                                  download=True)

    train_dataset, validation_dataset, test_dataset = get_stat_946_datasets(0.05, train_transform, test_transform)

elif args.dataset == 'svhn':
    num_classes = 10
    train_dataset = datasets.SVHN(root='data/',
                                  split='train',
                                  transform=train_transform,
                                  download=True)

    extra_dataset = datasets.SVHN(root='data/',
                                  split='extra',
                                  transform=train_transform,
                                  download=True)

    # Combine both training splits (https://arxiv.org/pdf/1605.07146.pdf)
    data = np.concatenate([train_dataset.data, extra_dataset.data], axis=0)
    labels = np.concatenate([train_dataset.labels, extra_dataset.labels], axis=0)
    train_dataset.data = data
    train_dataset.labels = labels

    test_dataset = datasets.SVHN(root='data/',
                                 split='test',
                                 transform=test_transform,
                                 download=True)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           pin_memory=True,
                                           num_workers=2)

validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=False,
                                          pin_memory=True,
                                          num_workers=2)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          shuffle=False,
                                          pin_memory=True,
                                          num_workers=2)

# ---------------------------------------------------------------------------------------
# Choose the Model
# ---------------------------------------------------------------------------------------
if args.model == 'resnet18':
    cnn = ResNet18(num_classes=num_classes)
elif args.model == 'wideresnet':
    if args.dataset == 'svhn':
        cnn = WideResNet(depth=16, num_classes=num_classes, widen_factor=8,
                         dropRate=0.4)
    else:
        cnn = WideResNet(depth=28, num_classes=num_classes, widen_factor=10,
                         dropRate=0.3)

cnn = cnn.cuda()
criterion = nn.CrossEntropyLoss().cuda()
cnn_optimizer = torch.optim.SGD(cnn.parameters(), lr=args.learning_rate,
                                momentum=0.9, nesterov=True, weight_decay=5e-4)

if args.dataset == 'svhn':
    scheduler = MultiStepLR(cnn_optimizer, milestones=[80, 120], gamma=0.1)
else:
    scheduler = MultiStepLR(cnn_optimizer, milestones=[60, 120, 160], gamma=0.2)


training_summary_file = os.path.join(RESULTS_DIR, test_id + '_training_summary.csv')
csv_logger = CSVLogger(args=args, fieldnames=['epoch', 'train_acc', 'test_acc'], filename=training_summary_file)


def test(loader):
    cnn.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0.
    total = 0.
    for images, labels in loader:
        images = images.cuda()
        labels = labels.cuda()

        with torch.no_grad():
            pred = cnn(images)

        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    val_acc = correct / total
    cnn.train()
    return val_acc


for epoch in range(args.epochs):

    xentropy_loss_avg = 0.
    correct = 0.
    total = 0.

    progress_bar = tqdm(train_loader)

    for i, (images, labels) in enumerate(progress_bar):
        progress_bar.set_description('Epoch ' + str(epoch))

        images = images.cuda()
        labels = labels.cuda()

        cnn.zero_grad()
        pred = cnn(images)

        xentropy_loss = criterion(pred, labels)
        xentropy_loss.backward()
        cnn_optimizer.step()

        xentropy_loss_avg += xentropy_loss.item()

        # Calculate running average of accuracy
        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels.data).sum().item()
        accuracy = correct / total

        progress_bar.set_postfix(
            xentropy='%.3f' % (xentropy_loss_avg / (i + 1)),
            acc='%.3f' % accuracy)

    test_acc = test(validation_loader)
    tqdm.write('test_acc: %.3f' % (test_acc))

    scheduler.step(epoch)

    row = {'epoch': str(epoch), 'train_acc': str(accuracy), 'test_acc': str(test_acc)}
    csv_logger.writerow(row)


model_weights_file = os.path.join(RESULTS_DIR, test_id + '_weights.pt')
torch.save(cnn.state_dict(), model_weights_file)
csv_logger.close()

# ---------------------------------------------------------------------------------------------
# Evaluate The test Data
# ---------------------------------------------------------------------------------------------
