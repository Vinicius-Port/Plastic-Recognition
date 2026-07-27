import os
import re
import json
import time
import argparse
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Set deterministic seed
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Detect Device
device = torch.device("cpu")
try:
    import torch_directml
    if torch_directml.is_available():
        device = torch_directml.device()
except ImportError:
    pass

if device.type == "cpu":
    if torch.cuda.is_available():
        device = torch.device("cuda")

# Dataset Class
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

POSSIBLE_DATA_DIRS = [
    "./Dataset_Wadaba",
    "../Datasets/Dataset_Wadaba",
    "../Dataset_Wadaba",
    r"C:\Users\Vinicius\Desktop\MestradoCodeAnti\ProjetoPlasticRecognition\Datasets\Dataset_Wadaba",
    "/kaggle/input/datasets/vinicius1portugal/dataset-wadaba/Dataset_modificado",
    "/kaggle/input/datasets/vinicius1portugal/dataset-wadaba",
    "/kaggle/input/dataset-wadaba/Dataset_Wadaba",
    "/kaggle/input/dataset-wadaba"
]

def resolve_dataset_dir(requested_dir=None):
    if requested_dir and os.path.exists(requested_dir):
        return requested_dir
    for cand in POSSIBLE_DATA_DIRS:
        if cand and os.path.exists(cand):
            return cand
    return requested_dir if requested_dir else "./Dataset_Wadaba"

def load_all_image_paths(data_dir):
    data_dir = resolve_dataset_dir(data_dir)
    class_names = ['Other', 'PET', 'PE_HD', 'PP', 'PS']
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    code_to_class = {'01': 'PET', '02': 'PE_HD', '05': 'PP', '06': 'PS', '07': 'Other'}
    
    img_paths, labels = [], []
    object_images = defaultdict(lambda: defaultdict(list))
    
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
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
                
    return img_paths, labels, object_images, class_names, class_to_idx

def get_transforms():
    IMG_SIZE = (224, 224)
    train_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
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

# LOOO Split Strategy
def create_datasets_looo(data_dir, val_split=0.2, seed=SEED):
    img_paths, labels, object_images, class_names, class_to_idx = load_all_image_paths(data_dir)
    train_paths, train_labels, val_paths, val_labels = [], [], [], []
    train_objs_all, val_objs_all = [], []
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
        
        train_objs_all.extend(train_objs)
        val_objs_all.extend(val_objs)
        
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
    
    split_info = {
        "strategy": "looo",
        "train_objects": train_objs_all,
        "val_objects": val_objs_all,
        "num_train_images": len(train_paths),
        "num_val_images": len(val_paths)
    }
    return train_ds, val_ds, class_names, split_info

# Random Split Strategy
def create_datasets_random(data_dir, val_split=0.2, seed=SEED):
    img_paths, labels, _, class_names, _ = load_all_image_paths(data_dir)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        img_paths, labels, test_size=val_split, random_state=seed, stratify=labels
    )
    train_tf, val_tf = get_transforms()
    train_ds = WaDaBaDataset(train_paths, train_labels, transform=train_tf)
    val_ds = WaDaBaDataset(val_paths, val_labels, transform=val_tf)
    split_info = {
        "strategy": "random",
        "num_train_images": len(train_paths),
        "num_val_images": len(val_paths)
    }
    return train_ds, val_ds, class_names, split_info

# Model Architectures
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

def build_model(arch, num_classes):
    if arch == "cnn":
        return CustomCNN(num_classes)
    elif arch == "resnet":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        for param in model.parameters(): param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif arch == "convnext":
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        for param in model.parameters(): param.requires_grad = False
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        return model
    elif arch == "swin":
        model = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
        for param in model.parameters(): param.requires_grad = False
        model.head = nn.Linear(model.head.in_features, num_classes)
        return model
    else:
        raise ValueError(f"Arquitetura desconhecida: {arch}")

# Training Loop with Checkpointing
def train_model_resilient(model, train_loader, val_loader, criterion, optimizer, epochs, device, output_dir, scheduler=None, patience=8):
    checkpoint_path = os.path.join(output_dir, "checkpoint.pth")
    history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}
    start_epoch = 0
    best_val_loss = float('inf')
    best_weights = None
    patience_counter = 0

    # Resume from checkpoint if present
    if os.path.exists(checkpoint_path):
        print(f"[CHECKPOINT] Carregando progresso salvo de: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        history = ckpt['history']
        best_val_loss = ckpt['best_val_loss']
        best_weights = ckpt['best_weights']
        patience_counter = ckpt['patience_counter']
        print(f"[CHECKPOINT] Retomando da Época {start_epoch + 1}/{epochs}")

    for epoch in range(start_epoch, epochs):
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

        # Validation
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

        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_accuracy'].append(epoch_val_acc)

        print(f"Epoch {epoch+1:02d}/{epochs:02d} - loss: {epoch_loss:.4f} - acc: {epoch_acc:.4f} - val_loss: {epoch_val_loss:.4f} - val_acc: {epoch_val_acc:.4f}")

        if scheduler:
            scheduler.step(epoch_val_loss)

        # Track Best Weights & Early Stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        # Save Epoch Checkpoint
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'history': history,
            'best_val_loss': best_val_loss,
            'best_weights': best_weights,
            'patience_counter': patience_counter
        }, checkpoint_path)

        if patience_counter >= patience:
            print(f"Early stopping ativado na época {epoch+1}!")
            if best_weights:
                model.load_state_dict(best_weights)
            break

    if best_weights:
        model.load_state_dict(best_weights)

    return history

# Save Metrics, Reports & Plots
def save_evaluation_outputs(model, val_loader, class_names, model_name, history, split_info, output_dir, device):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Plots
    epochs = range(1, len(history['accuracy']) + 1)
    plt.figure(figsize=(12, 4.5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['accuracy'], 'b-o', label='Acurácia Treino', markersize=4)
    plt.plot(epochs, history['val_accuracy'], 'r-o', label='Acurácia Validação', markersize=4)
    plt.title(f'Acurácia vs Épocas ({model_name})')
    plt.xlabel('Época'); plt.ylabel('Acurácia'); plt.grid(True, linestyle='--', alpha=0.6); plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['loss'], 'b-o', label='Perda (Loss) Treino', markersize=4)
    plt.plot(epochs, history['val_loss'], 'r-o', label='Perda (Loss) Validação', markersize=4)
    plt.title(f'Perda (Loss) vs Épocas ({model_name})')
    plt.xlabel('Época'); plt.ylabel('Perda (Loss)'); plt.grid(True, linestyle='--', alpha=0.6); plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "history_plot.png"), dpi=150)
    plt.close()

    # 2. Evaluation & Confusion Matrix
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs.to(device))
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    report_text = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    report_dict = classification_report(y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True)

    with open(os.path.join(output_dir, "metrics_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_text)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Previsto'); plt.ylabel('Real'); plt.title(f'Matriz de Confusão - {model_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
    plt.close()

    # 3. Model Weights (.pth)
    model_weights_path = os.path.join(output_dir, "model.pth")
    torch.save(model.state_dict(), model_weights_path)

    # 4. JSON Outputs (Split info, Metrics, Metadata)
    with open(os.path.join(output_dir, "split_info.json"), "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)

    metrics_json = {
        "model_name": model_name,
        "accuracy": float((y_true == y_pred).mean()),
        "weighted_f1": float(report_dict['weighted avg']['f1-score']),
        "macro_f1": float(report_dict['macro avg']['f1-score']),
        "classification_report": report_dict,
        "history": history
    }
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=2, ensure_ascii=False)

    metadata = {
        "model_name": model_name,
        "timestamp": datetime.datetime.now().isoformat(),
        "device": str(device),
        "seed": SEED,
        "epochs_run": len(history['accuracy'])
    }
    with open(os.path.join(output_dir, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n[SUCESSO] Todos os arquivos salvos em: {output_dir}")

def run_single_experiment(exp, data_dir, epochs, batch_size=32):
    name = exp["name"]
    split_type = exp["split"]
    arch = exp["arch"]
    output_dir = os.path.join("outputs", name)
    
    print("\n=======================================================")
    print(f"EXPERIMENTO: {name.upper()} (Divisão: {split_type.upper()} | Arquitetura: {arch.upper()})")
    print("=======================================================")

    if split_type == "looo":
        train_ds, val_ds, class_names, split_info = create_datasets_looo(data_dir)
    else:
        train_ds, val_ds, class_names, split_info = create_datasets_random(data_dir)

    num_classes = len(class_names)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    y_train = train_ds.labels
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float).to(device))

    model = build_model(arch, num_classes).to(device)
    if arch == "cnn":
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    else:
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4, weight_decay=0.01)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=3)

    history = train_model_resilient(
        model, train_loader, val_loader, criterion, optimizer,
        epochs=epochs, device=device, output_dir=output_dir, scheduler=scheduler, patience=8
    )

    save_evaluation_outputs(model, val_loader, class_names, name, history, split_info, output_dir, device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pipeline de Treinamento Resiliente e Modular")
    parser.add_argument("--data_dir", type=str, default="./Dataset_Wadaba", help="Caminho do dataset")
    parser.add_argument("--epochs", type=int, default=50, help="Número máximo de épocas")
    parser.add_argument("--model", type=str, default=None, help="Modelo específico (ex: modelo_resnet_looo, resnet_looo, ou 2)")
    
    args = parser.parse_args()

    experiments = [
        {"name": "modelo_cnn_looo", "split": "looo", "arch": "cnn"},
        {"name": "modelo_resnet_looo", "split": "looo", "arch": "resnet"},
        {"name": "modelo_convnext_looo", "split": "looo", "arch": "convnext"},
        {"name": "modelo_swin_looo", "split": "looo", "arch": "swin"},
        {"name": "modelo_cnn_random", "split": "random", "arch": "cnn"},
        {"name": "modelo_resnet_random", "split": "random", "arch": "resnet"},
        {"name": "modelo_convnext_random", "split": "random", "arch": "convnext"},
        {"name": "modelo_swin_random", "split": "random", "arch": "swin"}
    ]

    if args.model:
        target = args.model.lower().strip()
        filtered = []
        for i, exp in enumerate(experiments, 1):
            if target == str(i) or target == exp["name"] or target in exp["name"]:
                filtered.append(exp)
        if filtered:
            experiments = filtered
            print(f"[FILTRO] Treinando apenas: {[e['name'] for e in experiments]}")

    for exp in experiments:
        run_single_experiment(exp, args.data_dir, args.epochs)
