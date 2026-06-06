import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_loaders(batch_size=32):
    # Transformaciones básicas para normalizar las imágenes
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Descarga y carga de datos de entrenamiento y validación
    #Usamos CIFAR-10 como ejemplo, esta orientado a pc con pocos recursos
    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Configuración de DataLoaders optimizados para tu hardware local
    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True         # Eficiente para transferir datos a la GPU si está disponible
    )
    
    test_loader = DataLoader(
        test_set, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )

    return train_loader, test_loader

if __name__ == '__main__':
    train_loader, test_loader = get_loaders()
    print(f"DataLoader configurado. Lotes de entrenamiento: {len(train_loader)}")