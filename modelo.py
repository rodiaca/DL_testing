import torch
import torch.nn as nn
import torchvision.models as models

def get_model(num_classes=10):
    # Instanciamos ResNet-18 sin pesos preentrenados para entrenar desde cero
    model = models.resnet18(weights=None)
    
    # Adaptación para CIFAR-10:
    # La ResNet original espera imágenes de 224x224 y reduce el tamaño drásticamente al inicio.
    # Cambiamos la convolución inicial (7x7 con stride 2) por una de 3x3 con stride 1 para no perder información espacial.
    model.conv1 = nn.Conv2d(
        in_channels=3, 
        out_channels=64, 
        kernel_size=3, 
        stride=1, 
        padding=1, 
        bias=False
    )
    
    # Eliminamos la capa MaxPool inicial para preservar la resolución de 32x32 en las primeras etapas
    # Basicamente porque tenemos una resolución muy baja y no perder todos los datos
    model.maxpool = nn.Identity()
    
    # Adaptamos el clasificador final al número de clases del dataset
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model

if __name__ == '__main__':
    # Test rápido de dimensiones
    net = get_model()
    # Batch 32, Channels 3, H 32, W 32
    dummy_input = torch.randn(32, 3, 32, 32) # Creamos tensor de valores random (batch_size) = 32
    output = net(dummy_input)
    print(f"Forma del tensor de salida: {output.shape}") # Debe ser [32, 10] batch, clases