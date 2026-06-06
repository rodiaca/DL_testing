import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

#Queremos aumentar la precisión del modelo, para eso vamos a aumentar el dataset.

def get_loaders(batch_size=32, num_workers=4):

    # Low Accuracy Transformations (Baseline)
    # Transformaciones básicas para normalizar las imágenes
    # transform_lowAccuracy = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])
    # # Descarga y carga de datos de entrenamiento y validación
    # #Usamos CIFAR-10 como ejemplo, esta orientado a pc con pocos recursos
    # train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    ##############################################

    # 1. Transformaciones de ENTRENAMIENTO (Con Data Augmentation)
    # Imagenes rotadas y recortadas aleatoriamente. 
    # Cambio en la normalización para aumentar la diversidad de datos y mejorar la generalización del modelo.
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) # Medias/Desviaciones de CIFAR-10
    ])

    # 2. Transformaciones de TEST (Sin Augmentation, solo normalización)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Cargar datasets
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    # Configuración de DataLoaders optimizados para tu hardware local
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True         # Eficiente para transferir datos a la GPU si está disponible
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )

    return train_loader, test_loader

if __name__ == '__main__':
    train_loader, test_loader = get_loaders()
    print(f"DataLoader configurado. Lotes de entrenamiento: {len(train_loader)}")