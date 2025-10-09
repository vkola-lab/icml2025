import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import numpy as np
import torchvision
from medmnist import BloodMNIST  
from torchvision.datasets import ImageFolder
from fastai.vision.all import untar_data, URLs
import os

class CustomDataset(Dataset):
    # To be able to have multiple, i.e. 3, augmentations of each instance (augment your batch idea [1]) this function is written.
    # [1] Hoffer, Elad, et al. "Augment your batch: Improving generalization through instance repetition." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.
    def __init__(self, cifar_dataset, transform=None):
        self.cifar_dataset = cifar_dataset        
        self.transform = transform
        

    def __len__(self):
        return len(self.cifar_dataset)

    def __getitem__(self, idx):
        image, label = self.cifar_dataset[idx]
        augmented_images1 = self.transform(image)    
        augmented_images2 = self.transform(image) 
        augmented_images3 = self.transform(image)         
        return augmented_images1, augmented_images2, augmented_images3, label 
    
def get_cifar10():
    """CIFAR-10 transforms"""
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    
    # Transformations
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    dataset_path = "./datasets"
    np.random.seed(0)
    val_inds = np.sort(np.random.choice(50000, size=10000, replace=False))
    train_inds = np.setdiff1d(np.arange(50000), val_inds)

    dataset = torchvision.datasets.CIFAR10(root=dataset_path, train=True, download=True)
    trainset = torch.utils.data.Subset(dataset, train_inds)

    dataset_val = torchvision.datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=transform_test)
    valset = torch.utils.data.Subset(dataset_val, val_inds)

    train_dataset = CustomDataset(trainset, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8) #bs #32


    valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=False, num_workers=8)

    testset = torchvision.datasets.CIFAR10(root=dataset_path, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=8)

    pretrain_dataset = torchvision.datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=transform_train)
    pretrain_trainset = torch.utils.data.Subset(pretrain_dataset, train_inds)
    pretrain_loader = torch.utils.data.DataLoader(pretrain_trainset, batch_size=128, shuffle=True, num_workers=8)

    return pretrain_loader, trainloader, valloader, testloader

def get_cifar100():
    """CIFAR-100 transforms"""
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    
    # Transformations
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    dataset_path = "./datasets"
    np.random.seed(0)
    val_inds = np.sort(np.random.choice(50000, size=10000, replace=False))
    train_inds = np.setdiff1d(np.arange(50000), val_inds)

    dataset = torchvision.datasets.CIFAR100(root=dataset_path, train=True, download=True) #, transform=transform_train
    trainset = torch.utils.data.Subset(dataset, train_inds)

    dataset_val = torchvision.datasets.CIFAR100(root=dataset_path, train=True, download=True, transform=transform_test) #
    valset = torch.utils.data.Subset(dataset_val, val_inds)

    train_dataset = CustomDataset(trainset, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8) #bs #32


    valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=False, num_workers=8)

    pretrain_dataset = torchvision.datasets.CIFAR100(root=dataset_path, train=True, download=True, transform=transform_train) #, transform=transform_train
    pretrain_trainset = torch.utils.data.Subset(pretrain_dataset, train_inds)
    pretrain_loader = torch.utils.data.DataLoader(pretrain_trainset, batch_size=128, shuffle=True, num_workers=8)

    testset = torchvision.datasets.CIFAR100(root=dataset_path, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=8)

    return pretrain_loader, trainloader, valloader, testloader

def get_bloodmnist():
    """MNIST transforms"""
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(28, padding=2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    dataset_path = "./datasets"

    trainset = BloodMNIST(split="train", download=True, root=dataset_path)
    trainset.labels = trainset.labels.squeeze()
    train_dataset = CustomDataset(trainset, transform=transform_train)

    valset = BloodMNIST(split="val", download=True, transform=transform_test, root=dataset_path)
    valset.labels = valset.labels.squeeze()

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8)
    valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=False, num_workers=8)

    pretrain_trainset = BloodMNIST(root=dataset_path, split="train", download=True, transform=transform_train) 
    pretrain_trainset.labels = pretrain_trainset.labels.squeeze()
    pretrain_loader = torch.utils.data.DataLoader(pretrain_trainset, batch_size=512, shuffle=True, num_workers=8)

    testset = BloodMNIST(split="test", download=True, transform=transform_test, root=dataset_path)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=8)

    return pretrain_loader, trainloader, valloader, testloader


def get_imagenette():

    # For information about the dataset, see https://github.com/fastai/imagenette

    """ImageNet transforms"""
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    dataset_path = "./datasets/imagenette2-320/"
    if not os.path.exists(dataset_path):
        dataset_path = str(untar_data(URLs.IMAGENETTE_320, base=dataset_path))
        print(dataset_path)
    train_dataset_train_transforms = ImageFolder(dataset_path+'/train')
    train_dataset_all_len = len(train_dataset_train_transforms)

    np.random.seed(0)
    val_inds = np.sort(np.random.choice(train_dataset_all_len, size=int(train_dataset_all_len*0.1), replace=False))
    train_inds = np.setdiff1d(np.arange(train_dataset_all_len), val_inds)

    trainset = torch.utils.data.Subset(train_dataset_train_transforms, train_inds)
    train_dataset = CustomDataset(trainset, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8) 

    train_dataset_test_transforms = ImageFolder(dataset_path+'/train', transform_test)
    valset = torch.utils.data.Subset(train_dataset_test_transforms, val_inds)
    valloader = torch.utils.data.DataLoader(valset, batch_size=32, shuffle=False, num_workers=8)

    pretrain_dataset = ImageFolder(dataset_path+'/train', transform_train)
    pretrain_trainset = torch.utils.data.Subset(pretrain_dataset, train_inds)
    pretrain_loader = torch.utils.data.DataLoader(pretrain_trainset, batch_size=128, shuffle=True, num_workers=8)

    testset = ImageFolder(dataset_path+'/val', transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=8) 


    return pretrain_loader, trainloader, valloader, testloader

# Add custom dataset
"""
def get_custom():     
    return pretrain_loader, trainloader, valloader, testloader
""" 
# Transform registry for easy access
DATASET_FUNCTIONS = {
    'cifar10': get_cifar10,
    'cifar100': get_cifar100,
    'imagenette': get_imagenette,
    'bloodmnist': get_bloodmnist,
    #'custom': get_custom,
}

def get_dataset(dataset_name, **kwargs):
    """
    Main function to get transforms for any dataset
    
    Args:
        dataset_name: Name of the dataset ('cifar10', 'cifar100', 'imagenette', 'bloodmnist')
        **kwargs: Additional arguments passed to the specific transform function
    
    Returns:
        tuple: (pretrain_loader, trainloader, valloader, testloader)
    """
    if dataset_name not in DATASET_FUNCTIONS:
        available = list(DATASET_FUNCTIONS.keys())
        raise ValueError(f"Dataset '{dataset_name}' not supported. Available: {available}")
    
    return DATASET_FUNCTIONS[dataset_name](**kwargs)
