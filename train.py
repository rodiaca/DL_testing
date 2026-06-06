import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_loaders
from modelo import get_model

#evaluamos el modelo con el dataset de test para ver la precisión real después del entrenamiento
def evaluate(model, test_loader, device):
    model.eval() # Configura el modelo en modo de evaluación (desactiva Dropout/BatchNorm)
    correct = 0
    total = 0
    
    with torch.no_grad(): # Desactiva el cálculo de gradientes para ahorrar VRAM y acelerar computación
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            
            # Obtener el índice del valor máximo (la clase predicha)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
    accuracy = 100 * correct / total
    print(f"\n[EVALUACIÓN] Precisión en el dataset de Test: {accuracy:.2f}%\n")

def train():
    # 1. Configurar dispositivo (NVIDIA GTX 1650)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Entrenando en el dispositivo: {device}")

    # 2. Cargar datos y modelo (optimizados para baja VRAM)
    train_loader, test_loader = get_loaders(batch_size=32) # Tamaño de lote reducido para adaptarse a 4GB de VRAM
    model = get_model().to(device)               # Transferencia explícita a la GPU
    
    # 3. Función de pérdida y optimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Planificador para aumentar el accuracy: Reduce el LR a la mitad cada 2 épocas
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    NUM_EPOCHS = 5
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- ÉPOCA {epoch + 1}/{NUM_EPOCHS} ---")
        # 4. Bucle de entrenamiento (Epoch 1 para validar estabilidad)
        model.train()

        for batch_idx, (data, targets) in enumerate(train_loader):
            # Transferencia explícita de tensores Host -> Device 
            data, targets = data.to(device), targets.to(device)
        
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, targets)
        
            # Backward pass y optimización
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            if batch_idx % 400 == 0:
                print(f"Lote [{batch_idx}/{len(train_loader)}] | Pérdida: {loss.item():.4f}")
                # Estrategia proactiva: Vaciar caché de PyTorch para mitigar fragmentación en 4GB [cite: 26]
                torch.cuda.empty_cache()
        
        # Actualizamos el learning rate al final de la época
        scheduler.step()

        # Al terminar la época, evaluamos el rendimiento real
        evaluate(model, test_loader, device)

if __name__ == '__main__':
    train()