import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from collections import defaultdict

import argparse

# =====================================================================
# Configurações do Experimento
# =====================================================================
parser = argparse.ArgumentParser(description="Treinamento Benchmark de 8 Modelos (LOOO vs Random Split)")
parser.add_argument("--data_dir", type=str, default=None, help="Caminho para o diretório do dataset")
parser.add_argument("--epochs", type=int, default=50, help="Número de épocas por modelo")
args, _ = parser.parse_known_args()

POSSIBLE_DATA_DIRS = [
    args.data_dir,
    "/kaggle/input/datasets/vinicius1portugal/dataset-wadaba/Dataset_modificado",
    "/kaggle/input/datasets/vinicius1portugal/dataset-wadaba",
    "./Dataset_Wadaba",
    "../Dataset_Wadaba",
    "../Datasets/Dataset_Wadaba",
    "./Datasets/Dataset_Wadaba",
    "/kaggle/input/dataset-wadaba/Dataset_Wadaba",
    "/kaggle/input/dataset-wadaba",
    "/kaggle/input/wadaba/Dataset_Wadaba",
    "/kaggle/input/wadaba",
    "/content/Dataset_Wadaba",
    "/content/drive/MyDrive/Dataset_Wadaba",
    r"C:\Users\Vinicius\Desktop\MestradoCodeAnti\ProjetoPlasticRecognition\Datasets\Dataset_Wadaba"
]

DATA_DIR = None
for candidate in POSSIBLE_DATA_DIRS:
    if candidate and os.path.exists(candidate):
        # Verifica se realmente contém imagens
        has_imgs = False
        for root, _, files in os.walk(candidate):
            if any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in files):
                has_imgs = True
                break
        if has_imgs:
            DATA_DIR = candidate
            break

if DATA_DIR is None:
    DATA_DIR = "./Dataset_Wadaba"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = args.epochs

# Configuração de dispositivo (CUDA / DirectML / CPU)
device = torch.device("cpu")
try:
    import torch_directml
    if torch_directml.is_available():
        device = torch_directml.device()
        print(f"[INFO] Dispositivo Ativo: GPU via DirectML ({torch_directml.device_name(0)})")
except ImportError:
    pass

if device.type == "cpu":
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] Dispositivo Ativo: GPU via CUDA ({torch.cuda.get_device_name(0)})")
    else:
        print("[INFO] Dispositivo Ativo: CPU")

# =====================================================================
# Custom Dataset e Funções de Carregamento
# =====================================================================
class WaDaBaDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, torch.tensor(label, dtype=torch.long)

def load_all_image_paths(data_dir):
    class_names = ['Other', 'PET', 'PE_HD', 'PP', 'PS']
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    code_to_class = {'01': 'PET', '02': 'PE_HD', '05': 'PP', '06': 'PS', '07': 'Other'}
    
    img_paths, labels = [], []
    object_images = defaultdict(lambda: defaultdict(list))
    
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(root, f)
                filename = os.path.basename(path)
                parent_dir = os.path.basename(os.path.dirname(path))
                
                if parent_dir in class_to_idx:
                    class_name = parent_dir
                else:
                    match_class = re.search(r'_a(\d{2})', filename)
                    if match_class:
                        class_code = match_class.group(1)
                        class_name = code_to_class.get(class_code, 'Other')
                    else:
                        class_name = 'Other'
                        
                match_obj = re.match(r'^(\d+)', filename)
                obj_id = match_obj.group(1) if match_obj else filename
                
                object_images[class_name][obj_id].append(path)
                img_paths.append(path)
                labels.append(class_to_idx[class_name])
                
    if len(img_paths) == 0:
        print(f"\n[ERRO] Nenhum arquivo de imagem (.jpg, .png) foi encontrado em: {data_dir}")
        print("Por favor, verifique se a pasta do dataset foi descompactada e se o caminho está correto!")
        raise FileNotFoundError(f"Diretório de dataset vazio ou não encontrado: {data_dir}")

    print(f"[INFO] Dataset carregado de '{data_dir}': Total de {len(img_paths)} imagens encontradas.")
    return img_paths, labels, object_images, class_names, class_to_idx

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(72),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.2), ratio=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform

# --- Estratégia 1: Leave-One-Object-Out (LOOO / Group Split) ---
def create_datasets_looo(data_dir, val_split=0.2, seed=123):
    print("\n--- Carregando dados com estratégia LOOO (Group Split por Objeto) ---")
    img_paths, labels, object_images, class_names, class_to_idx = load_all_image_paths(data_dir)
    train_paths, train_labels, val_paths, val_labels = [], [], [], []
    rng = np.random.default_rng(seed)
    
    for class_name in class_names:
        class_idx = class_to_idx[class_name]
        objects_dict = object_images[class_name]
        unique_obj_ids = sorted(list(objects_dict.keys()))
        if not unique_obj_ids:
            continue
        shuffled_objs = unique_obj_ids.copy()
        rng.shuffle(shuffled_objs)
        
        split_idx = int(len(shuffled_objs) * (1.0 - val_split))
        if split_idx == len(shuffled_objs) and len(shuffled_objs) > 1:
            split_idx -= 1
        if split_idx == 0 and len(shuffled_objs) > 0:
            split_idx = 1
            
        train_objs = shuffled_objs[:split_idx]
        val_objs = shuffled_objs[split_idx:]
        
        for obj_id in train_objs:
            for path in objects_dict[obj_id]:
                train_paths.append(path)
                train_labels.append(class_idx)
        for obj_id in val_objs:
            for path in objects_dict[obj_id]:
                val_paths.append(path)
                val_labels.append(class_idx)
                
    train_tf, val_tf = get_transforms()
    train_ds = WaDaBaDataset(train_paths, train_labels, transform=train_tf)
    val_ds = WaDaBaDataset(val_paths, val_labels, transform=val_tf)
    return train_ds, val_ds, class_names

# --- Estratégia 2: Separação Aleatória Estratificada (Random Split por Imagem) ---
def create_datasets_random(data_dir, val_split=0.2, seed=123):
    print("\n--- Carregando dados com estratégia RANDOM (Separação Aleatória por Imagem) ---")
    img_paths, labels, _, class_names, _ = load_all_image_paths(data_dir)
    
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        img_paths, labels, test_size=val_split, random_state=seed, stratify=labels
    )
    
    train_tf, val_tf = get_transforms()
    train_ds = WaDaBaDataset(train_paths, train_labels, transform=train_tf)
    val_ds = WaDaBaDataset(val_paths, val_labels, transform=val_tf)
    return train_ds, val_ds, class_names

# =====================================================================
# Arquiteturas de Modelos
# =====================================================================
class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.5), nn.Linear(256 * 14 * 14, 128), nn.ReLU(), nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

def build_resnet_model(num_classes):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def build_convnext_model(num_classes):
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    return model

def build_swin_model(num_classes):
    model = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model

# =====================================================================
# Callbacks de Treinamento
# =====================================================================
class History:
    def __init__(self):
        self.history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}

class EarlyStopping:
    def __init__(self, patience=8, restore_best_weights=True):
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_weights = None
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif val_loss >= self.best_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
    def restore(self, model):
        if self.restore_best_weights and self.best_weights is not None:
            model.load_state_dict(self.best_weights)

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device, early_stopping=None, scheduler=None):
    history = History()
    for epoch in range(epochs):
        model.train()
        running_loss, running_corrects, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels)
            total += inputs.size(0)
        epoch_loss = running_loss / total
        epoch_acc = (running_corrects.double() / total).item()
        model.eval()
        val_loss, val_corrects, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels)
                val_total += inputs.size(0)
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = (val_corrects.double() / val_total).item()
        history.history['loss'].append(epoch_loss)
        history.history['accuracy'].append(epoch_acc)
        history.history['val_loss'].append(epoch_val_loss)
        history.history['val_accuracy'].append(epoch_val_acc)
        print(f"Epoch {epoch+1:02d}/{epochs:02d} - loss: {epoch_loss:.4f} - acc: {epoch_acc:.4f} - val_loss: {epoch_val_loss:.4f} - val_acc: {epoch_val_acc:.4f}")
        if scheduler:
            scheduler.step(epoch_val_loss)
        if early_stopping:
            early_stopping(epoch_val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping ativado!")
                early_stopping.restore(model)
                break
    return history

def plot_history(history, model_name):
    epochs = range(1, len(history.history['accuracy']) + 1)
    
    plt.figure(figsize=(12, 4.5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history['accuracy'], 'b-o', label='Acurácia Treino')
    plt.plot(epochs, history.history['val_accuracy'], 'r-o', label='Acurácia Validação')
    plt.title(f'Acurácia vs Épocas ({model_name})')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history['loss'], 'b-o', label='Perda (Loss) Treino')
    plt.plot(epochs, history.history['val_loss'], 'r-o', label='Perda (Loss) Validação')
    plt.title(f'Perda (Loss) vs Épocas ({model_name})')
    plt.xlabel('Época')
    plt.ylabel('Perda (Loss)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{model_name}_history.png")
    plt.show()
    plt.close()

def evaluate_model(model, val_loader, class_names, model_name, device):
    print(f"\n--- Avaliação Detalhada: {model_name} ---")
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    report = classification_report(y_true, y_pred, labels=list(range(len(class_names))), target_names=class_names, zero_division=0)
    print(report)
    
    with open(f"{model_name}_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
        
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Previsto')
    plt.ylabel('Real')
    plt.title(f'Matriz de Confusão - {model_name}')
    plt.tight_layout()
    plt.savefig(f"{model_name}_confusion_matrix.png")
    plt.show()
    plt.close()
    
    acc = (y_true == y_pred).mean()
    return acc, report

# =====================================================================
# Fluxo Principal de Treinamento dos 8 Modelos
# =====================================================================
def run_experiments():
    results = {}
    
    # Lista dos 8 experimentos
    experiments = [
        # --- Grupo 1: LOOO (Leave-One-Object-Out) ---
        {"name": "modelo_cnn_looo", "split": "looo", "arch": "cnn"},
        {"name": "modelo_resnet_looo", "split": "looo", "arch": "resnet"},
        {"name": "modelo_convnext_looo", "split": "looo", "arch": "convnext"},
        {"name": "modelo_swin_looo", "split": "looo", "arch": "swin"},
        
        # --- Grupo 2: RANDOM (Separação Aleatória por Imagem) ---
        {"name": "modelo_cnn_random", "split": "random", "arch": "cnn"},
        {"name": "modelo_resnet_random", "split": "random", "arch": "resnet"},
        {"name": "modelo_convnext_random", "split": "random", "arch": "convnext"},
        {"name": "modelo_swin_random", "split": "random", "arch": "swin"},
    ]
    
    for exp in experiments:
        name = exp["name"]
        split_type = exp["split"]
        arch = exp["arch"]
        weights_filename = f"{name}.pth"
        
        print("\n=======================================================")
        print(f"EXPERIMENTO: {name.upper()} (Divisão: {split_type.upper()} | Arquitetura: {arch.upper()})")
        print("=======================================================")
        
        if os.path.exists(weights_filename):
            print(f"[INFO] Arquivo {weights_filename} já existe. Pulando treinamento...")
            continue
            
        # 1. Carregar dataset com o split correspondente
        if split_type == "looo":
            train_ds, val_ds, class_names = create_datasets_looo(DATA_DIR)
        else:
            train_ds, val_ds, class_names = create_datasets_random(DATA_DIR)
            
        num_classes = len(class_names)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
        # Pesos das classes
        y_train = train_ds.labels
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
        
        # 2. Instanciar Modelo
        if arch == "cnn":
            model = CustomCNN(num_classes).to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
        elif arch == "resnet":
            model = build_resnet_model(num_classes).to(device)
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, weight_decay=0.01)
        elif arch == "convnext":
            model = build_convnext_model(num_classes).to(device)
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4, weight_decay=0.01)
        elif arch == "swin":
            model = build_swin_model(num_classes).to(device)
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4, weight_decay=0.01)
            
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        early_stopping = EarlyStopping(patience=8, restore_best_weights=True)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=3, min_lr=1e-6)
        
        # 3. Treinar
        history = train_model(
            model, train_loader, val_loader, criterion, optimizer,
            epochs=EPOCHS, device=device, early_stopping=early_stopping, scheduler=scheduler
        )
        
        # 4. Salvar pesos, gerar gráficos e avaliar
        torch.save(model.state_dict(), weights_filename)
        plot_history(history, name)
        acc, report = evaluate_model(model, val_loader, class_names, name, device)
        results[name] = acc
        
    print("\n=======================================================")
    print("RESUMO COMPARATIVO DOS 8 MODELOS")
    print("=======================================================")
    for model_name, acc in results.items():
        print(f"{model_name}: Acurácia de Validação = {acc*100:.2f}%")

if __name__ == '__main__':
    run_experiments()
