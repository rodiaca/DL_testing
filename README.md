# CIFAR-10 Verification & Deep Learning Pipeline

Este módulo contiene el pipeline de validación y entrenamiento sobre el dataset CIFAR-10, diseñado como paso previo e indicador de estabilidad antes del despliegue de kernels CUDA nativos. El pipeline está optimizado para entornos locales con restricciones de VRAM (arquitecturas de 4 GB).

## Arquitectura del Modelo
Se implementa una **ResNet-18** modificada específicamente para imágenes de baja resolución ($32\times32$ píxeles):
* **`conv1` adaptada:** Reducida de $7\times7$ (stride 2) a $3\times3$ (stride 1) con `padding=1` para preservar la información espacial inicial y evitar submuestreo agresivo.
* **Bypass de MaxPool:** Capa `maxpool` sustituida por `nn.Identity()` para mantener las dimensiones estructurales críticas en las primeras capas del grafo.

## Optimización de Hardware Local
* **Data Loaders:** Configuración de `num_workers=4` para balancear la carga multitarea sin caer en *over-subscription* de hilos en CPU. Uso de `pin_memory=True` para acelerar transferencias síncronas Host-to-Device a través de PCIe.
* **Gestión de VRAM:** Implementación de vaciado de caché proactivo mediante `torch.cuda.empty_cache()` en el bucle principal para mitigar la fragmentación en entornos de 4 GB.

## Estructura del Proyecto
* `dataset.py`: Pipeline de datos con transformaciones en línea (Data Augmentation con *RandomCrop* y *RandomHorizontalFlip*) y normalización estadística de CIFAR-10.
* `model.py`: Definición de la red y script de validación de dimensiones sintéticas (*dummy inputs*).
* `train.py`: Bucle de entrenamiento principal, planificación de Learning Rate (`StepLR`) y módulo de evaluación síncrona en el conjunto de test.

## 🔧 Ejecución
1. Activar el entorno virtual:
   ```bash
   source ~/dl_env/bin/activate