# INF2008 - Keypoint Detection (Vertebra C2/C4)

Detecção de keypoints (C2 e C4) em vértebras cervicais usando Keypoint R-CNN com PyTorch.

## Requisitos

- Python 3.10+
- CUDA (recomendado para treinamento)
- Conda ou venv

## Instalação

1. Clone o repositório:
```bash
git clone https://github.com/marcellonascif/INF2008-keypoints.git
cd INF2008-keypoints
```

2. Instale as dependências:
```bash
pip install torch torchvision opencv-python albumentations pycocotools matplotlib jupyter
```

## Estrutura do Dataset

```
dataset/
├── fold1/
│   ├── treino/
│   │   ├── images/
│   │   └── labels/  # arquivos .json
│   └── teste/
│       ├── images/
│       └── labels/
├── fold2/
└── fold3/
```

## Como Rodar

### 1. Treinamento

Abra o notebook `src/train_keypoints.ipynb` e configure:

```python
FOLD_NUMBER = 1          # Escolha o fold (1, 2 ou 3)
USE_AUGMENTATION = False # True para usar data augmentation
num_epochs = 50          # Número de épocas
```

Execute todas as células. O modelo será salvo em:
```
models/fold{N}_{aug_type}/keypointsrcnn_weights_fold{N}_{aug_type}.pth
```

### 2. Carregar o Modelo

Para carregar o modelo treinado:

```python
from src.train_keypoints import get_model

# Carregue o modelo
model = get_model(num_keypoints=2,
                  weights_path='models/fold3_no_aug/keypointsrcnn_weights_fold3_no_aug.pth')
model.eval()

```

## Métricas

O modelo é avaliado usando:
- **Average Precision (AP)** - Para bbox e keypoints
- **Average Recall (AR)** - Para bbox e keypoints
- **F1 Score** - Harmônica entre AP e AR

## Arquivos

- `src/train_keypoints.ipynb` - Notebook principal de treinamento
- `src/detection/` - Utilitários do PyTorch para detecção
- `models/` - Modelos treinados
- `dataset/` - Dataset com os folds
