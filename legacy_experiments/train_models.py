import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from collections import defaultdict

# Configurações com busca dinâmica do diretório do dataset
POSSIBLE_DATA_DIRS = [
    "./Dataset_Wadaba",
    "../Dataset_Wadaba",
    "../Datasets/Dataset_Wadaba",
    "./Datasets/Dataset_Wadaba",
    r"C:\Users\Vinicius\Desktop\MestradoCodeAnti\ProjetoPlasticRecognition\Datasets\Dataset_Wadaba"
]

DATA_DIR = None
for candidate in POSSIBLE_DATA_DIRS:
    if os.path.exists(candidate):
        DATA_DIR = candidate
        break

if DATA_DIR is None:
    DATA_DIR = "./Dataset_Wadaba"  # Padrão se o usuário descompactar no diretório atual

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50

# Configuração de dispositivo com suporte a AMD GPU via DirectML
device = torch.device("cpu")
try:
    import torch_directml
    if torch_directml.is_available():
        device = torch_directml.device()
        print(f"Utilizando GPU via DirectML: {torch_directml.device_name(0)}")
except ImportError:
    pass

if device.type == "cpu":
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Utilizando GPU via CUDA (NVIDIA).")
    else:
        print("Utilizando a CPU para o treinamento.")

# --- Custom Dataset ---
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

# --- Carregar e dividir dados usando LOOO ---
def create_datasets(data_dir, val_split=0.2, seed=123):
    print("Carregando datasets com estratégia Leave-One-Object-Out (Group Split)...")
    
    class_names = ['Other', 'PET', 'PE_HD', 'PP', 'PS']
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    
    # Mapeamento do código da base (aXX) para o nome da classe
    code_to_class = {
        '01': 'PET',
        '02': 'PE_HD',
        '05': 'PP',
        '06': 'PS',
        '07': 'Other'
    }
    
    # 1. Buscar recursivamente todas as imagens
    img_paths = []
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_paths.append(os.path.join(root, f))
                
    print(f"Total de arquivos de imagem encontrados: {len(img_paths)}")
    
    # 2. Agrupar caminhos de imagem por classe e ID do objeto
    object_images = defaultdict(lambda: defaultdict(list))
    
    for path in img_paths:
        filename = os.path.basename(path)
        parent_dir = os.path.basename(os.path.dirname(path))
        
        # Determinar a classe
        if parent_dir in class_to_idx:
            class_name = parent_dir
        else:
            match_class = re.search(r'_a(\d{2})', filename)
            if match_class:
                class_code = match_class.group(1)
                class_name = code_to_class.get(class_code, 'Other')
            else:
                class_name = 'Other'
                
        # Determinar o ID do objeto
        match_obj = re.match(r'^(\d+)', filename)
        if match_obj:
            obj_id = match_obj.group(1)
        else:
            obj_id = filename
            
        object_images[class_name][obj_id].append(path)
        
    # 3. Divisão de treino e validação por ID de objeto (evita data leakage)
    train_paths = []
    train_labels = []
    val_paths = []
    val_labels = []
    
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
                
    print(f"Divisão concluída com sucesso!")
    print(f"Classes encontradas: {class_names}")
    print(f"Total de imagens de Treino: {len(train_paths)}")
    print(f"Total de imagens de Validação: {len(val_paths)}")
    
    # 4. Definir transformações (Data Augmentation melhorado na V2)
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
    
    train_ds = WaDaBaDataset(train_paths, train_labels, transform=train_transform)
    val_ds = WaDaBaDataset(val_paths, val_labels, transform=val_transform)
    
    return train_ds, val_ds, class_names

# --- Modelos ---
class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256 * 14 * 14, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def build_transfer_learning_model(num_classes):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def build_convnext_model(num_classes):
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    return model

def build_swin_model(num_classes):
    model = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)
    return model

# --- Histórico e Callbacks auxiliares ---
class History:
    def __init__(self):
        self.history = {
            'accuracy': [],
            'val_accuracy': [],
            'loss': [],
            'val_loss': []
        }

class EarlyStopping:
    def __init__(self, patience=6, restore_best_weights=True):
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            # Clona os pesos na CPU para evitar que fiquem corrompidos se a GPU for suspensa
            self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif val_loss >= self.best_loss:
            self.counter += 1
            print(f"EarlyStopping contador: {self.counter} de {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
            
    def restore(self, model):
        if self.restore_best_weights and self.best_weights is not None:
            try:
                model.load_state_dict(self.best_weights)
            except Exception as e:
                print(f"[AVISO] Falha ao carregar pesos no dispositivo atual: {str(e)}")
                print("Forçando modelo para CPU e restaurando pesos lá...")
                model.cpu()
                model.load_state_dict(self.best_weights)
            print("Pesos do melhor modelo restaurados.")

# --- Funções de Treino e Validação ---
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device, early_stopping=None, scheduler=None):
    history = History()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
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
        
        # Validação
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
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
        
        print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f} - accuracy: {epoch_acc:.4f} - val_loss: {epoch_val_loss:.4f} - val_accuracy: {epoch_val_acc:.4f}")
        
        if scheduler:
            scheduler.step(epoch_val_loss)
            
        if early_stopping:
            early_stopping(epoch_val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping ativado!")
                early_stopping.restore(model)
                break
                
    return history

def plot_history(history, title):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Treino Acc')
    plt.plot(epochs_range, val_acc, label='Validação Acc')
    plt.legend(loc='lower right')
    plt.title(f'Acurácia ({title})')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Treino Loss')
    plt.plot(epochs_range, val_loss, label='Validação Loss')
    plt.legend(loc='upper right')
    plt.title(f'Loss / Perda ({title})')

    plt.savefig(f'{title}_v2_history.png')
    plt.close()

def evaluate_model(model, val_loader, class_names, title, device):
    print(f"\n--- Avaliação Detalhada: {title} ---")
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Corrigido na V2.1: Especifica os 'labels' para evitar erros quando algumas classes não estão presentes na validação
    report = classification_report(y_true, y_pred, labels=list(range(len(class_names))), target_names=class_names, zero_division=0)
    print("Relatório de Classificação:\n", report)

    with open(f"{title}_v2_report.txt", "w") as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Previsto')
    plt.ylabel('Real')
    plt.title(f'Matriz de Confusão - {title}')
    plt.savefig(f'{title}_v2_confusion_matrix.png')
    plt.close()

# --- Driver Principal ---
def main():
    if not os.path.exists(DATA_DIR):
        print(f"ERRO: Diretório de dados não encontrado em {DATA_DIR}")
        return

    train_ds, val_ds, class_names = create_datasets(DATA_DIR)
    num_classes = len(class_names)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Pesos de classe para tratar o desbalanceamento
    y_train = train_ds.labels
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"Pesos das classes calculados: {class_weights}")

    # ==============================
    # TREINANDO MODELO 1: CNN Customizada (V2)
    # ==============================
    if os.path.exists('modelo_cnn_v2.pth'):
        print("\n[INFO] Modelo CNN V2 já treinado. Pulando treinamento...")
    else:
        print("\n==============================")
        print("TREINANDO MODELO 1: CNN Customizada (V2)")
        print("==============================")
        cnn_model = CustomCNN(num_classes).to(device)
        cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
        cnn_criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        
        cnn_early_stopping = EarlyStopping(patience=8, restore_best_weights=True)
        cnn_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            cnn_optimizer, mode='min', factor=0.3, patience=3, min_lr=1e-6
        )
        
        cnn_history = train_model(
            cnn_model, train_loader, val_loader, cnn_criterion, cnn_optimizer,
            epochs=EPOCHS, device=device, early_stopping=cnn_early_stopping, scheduler=cnn_scheduler
        )
        
        torch.save(cnn_model.state_dict(), 'modelo_cnn_v2.pth')
        plot_history(cnn_history, 'CNN')
        evaluate_model(cnn_model, val_loader, class_names, 'CNN', device)

    # ==============================
    # TREINANDO MODELO 2: Transfer Learning (ResNet-50 V2)
    # ==============================
    if os.path.exists('modelo_transfer_v2.pth'):
        print("\n[INFO] Modelo Transfer Learning (ResNet-50) V2 já treinado. Pulando treinamento...")
    else:
        print("\n==============================")
        print("TREINANDO MODELO 2: Transfer Learning (ResNet-50 V2)")
        print("==============================")
        tl_model = build_transfer_learning_model(num_classes).to(device)
        tl_criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        
        # Fase 1: Treinando apenas a camada do topo (fc)
        print("\n--- Fase 1: Treinando a camada do topo (fc) ---")
        tl_optimizer = optim.Adam(filter(lambda p: p.requires_grad, tl_model.parameters()), lr=0.001)
        
        tl_early_stopping1 = EarlyStopping(patience=8, restore_best_weights=True)
        tl_scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(
            tl_optimizer, mode='min', factor=0.3, patience=3, min_lr=1e-6
        )
        
        tl_history_phase1 = train_model(
            tl_model, train_loader, val_loader, tl_criterion, tl_optimizer,
            epochs=10, device=device, early_stopping=tl_early_stopping1, scheduler=tl_scheduler1
        )

        # Fase 2: Fine-Tuning do modelo base (descongelando o último bloco residual 'layer4')
        print("\n--- Fase 2: Fine-Tuning do modelo base (layer4) ---")
        for param in tl_model.layer4.parameters():
            param.requires_grad = True
            
        tl_optimizer2 = optim.Adam(filter(lambda p: p.requires_grad, tl_model.parameters()), lr=1e-5)
        tl_early_stopping2 = EarlyStopping(patience=8, restore_best_weights=True)
        tl_scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(
            tl_optimizer2, mode='min', factor=0.3, patience=3, min_lr=1e-6
        )
        
        tl_history_phase2 = train_model(
            tl_model, train_loader, val_loader, tl_criterion, tl_optimizer2,
            epochs=EPOCHS, device=device, early_stopping=tl_early_stopping2, scheduler=tl_scheduler2
        )

        # Unificar históricos
        for key in tl_history_phase1.history.keys():
            tl_history_phase1.history[key].extend(tl_history_phase2.history[key])

        torch.save(tl_model.state_dict(), 'modelo_transfer_v2.pth')
        plot_history(tl_history_phase1, 'TransferLearning')
        evaluate_model(tl_model, val_loader, class_names, 'TransferLearning', device)

    # ==============================
    # TREINANDO MODELO 3: Modern SOTA CNN (ConvNeXt-Tiny)
    # ==============================
    if os.path.exists('modelo_convnext_v2.pth'):
        print("\n[INFO] Modelo ConvNeXt-Tiny V2 já treinado. Pulando treinamento...")
    else:
        print("\n==============================")
        print("TREINANDO MODELO 3: ConvNeXt-Tiny (Modern SOTA CNN)")
        print("==============================")
        convnext_model = build_convnext_model(num_classes).to(device)
        convnext_criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        
        print("\n--- Treinando a cabeça de classificação com AdamW (50 Épocas na GPU) ---")
        convnext_optimizer = optim.AdamW(filter(lambda p: p.requires_grad, convnext_model.parameters()), lr=5e-4, weight_decay=0.01)
        convnext_early_stopping = EarlyStopping(patience=8, restore_best_weights=True)
        convnext_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            convnext_optimizer, mode='min', factor=0.3, patience=3, min_lr=1e-6
        )
        
        convnext_history = train_model(
            convnext_model, train_loader, val_loader, convnext_criterion, convnext_optimizer,
            epochs=EPOCHS, device=device, early_stopping=convnext_early_stopping, scheduler=convnext_scheduler
        )

        torch.save(convnext_model.state_dict(), 'modelo_convnext_v2.pth')
        plot_history(convnext_history, 'ConvNeXt')
        evaluate_model(convnext_model, val_loader, class_names, 'ConvNeXt', device)

    # ==============================
    # TREINANDO MODELO 4: Modern SOTA Transformer (Swin Transformer Tiny)
    # ==============================
    if os.path.exists('modelo_swin_v2.pth'):
        print("\n[INFO] Modelo Swin Transformer V2 já treinado. Pulando treinamento...")
    else:
        print("\n==============================")
        print("TREINANDO MODELO 4: Swin Transformer Tiny (Swin-T)")
        print("==============================")
        swin_model = build_swin_model(num_classes).to(device)
        swin_criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        
        print("\n--- Treinando a cabeça de classificação com AdamW (50 Épocas na GPU) ---")
        swin_optimizer = optim.AdamW(filter(lambda p: p.requires_grad, swin_model.parameters()), lr=5e-4, weight_decay=0.01)
        swin_early_stopping = EarlyStopping(patience=8, restore_best_weights=True)
        swin_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            swin_optimizer, mode='min', factor=0.3, patience=3, min_lr=1e-6
        )
        
        swin_history = train_model(
            swin_model, train_loader, val_loader, swin_criterion, swin_optimizer,
            epochs=EPOCHS, device=device, early_stopping=swin_early_stopping, scheduler=swin_scheduler
        )

        torch.save(swin_model.state_dict(), 'modelo_swin_v2.pth')
        plot_history(swin_history, 'Swin')
        evaluate_model(swin_model, val_loader, class_names, 'Swin', device)

    print("\nTreinamento completo das 4 arquiteturas concluído com sucesso!")
    print("Os modelos (.pth) e relatórios foram gerados no diretório atual.")

if __name__ == '__main__':
    main()
