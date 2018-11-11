# ----------------------------------------------------------------------------------------------------
# Reference https://github.com/uoguelph-mlrg/Cutout
# Modified for the Stat946 Data Challenge
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
from datetime import datetime

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
dataset_options = ['cifar100']


BASE_RESULTS_DIR = './results'
if not os.path.exists(BASE_RESULTS_DIR):
    os.mkdir(BASE_RESULTS_DIR)

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

# print the input arguments:
for k, v in args.__dict__.items():
    print("{}: {}".format(k, v))

# Initialization  -------------------------------------------------
test_id = args.dataset + '_' + args.model

results_dir = os.path.join(BASE_RESULTS_DIR, test_id)
if not os.path.exists(results_dir):
    os.mkdir(results_dir)

training_summary_file = os.path.join(results_dir, 'training_summary.csv')
model_weights_file = os.path.join(results_dir, 'weights.pt')
predictions_file = os.path.join(results_dir, 'predictions.csv')


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
    test_ds = Stat946TestDataSet(x_test, test_preprocessing)

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
train_validation_split = 0.05

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

train_dataset, validation_dataset, test_dataset = get_stat_946_datasets(
    train_validation_split, train_transform, test_transform)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=2)

validation_loader = torch.utils.data.DataLoader(
    dataset=validation_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    pin_memory=True,
    num_workers=2)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
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


csv_logger = CSVLogger(args=args, fieldnames=['epoch', 'train_acc', 'test_acc'], filename=training_summary_file)

# ---------------------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------------------
print("Training Started {}".format('*'*80))
print("Train/Validation Split={}. (nTrain {}, nValidation {})".format(
    train_validation_split,
    train_dataset.__len__(),
    validation_dataset.__len__()))

start_time = datetime.now()


def validation_error(loader):
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

    test_acc = validation_error(validation_loader)
    tqdm.write('test_acc: %.3f' % test_acc)

    scheduler.step(epoch)

    row = {'epoch': str(epoch), 'train_acc': str(accuracy), 'test_acc': str(test_acc)}
    csv_logger.writerow(row)

torch.save(cnn.state_dict(), model_weights_file)
csv_logger.close()
print("Training took {}".format(datetime.now() - start_time))

# ---------------------------------------------------------------------------------------------
# Evaluate The test Data
# ---------------------------------------------------------------------------------------------
print("Evaluating Test Data {}".format('*'*80))
predictions_csv_logger = CSVLogger(fieldnames=['ids', 'labels'], filename=predictions_file)

cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
for idx, test_image in enumerate(test_loader):

    test_image = test_image.cuda()

    with torch.no_grad():
        pred = cnn(test_image)
        max_pred = torch.max(pred, 1)[1]

        row = {'ids': str(idx), 'labels': str(np.int(max_pred))}
        predictions_csv_logger.writerow(row)

predictions_csv_logger.close()
