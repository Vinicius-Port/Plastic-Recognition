import os
import cv2
import re
import shutil
import hashlib
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import models, transforms

# ---------------------------------------------------------
# DEFINIÇÃO DAS ARQUITETURAS IGUAIS AO TREINAMENTO
# ---------------------------------------------------------
class CustomCNN(nn.Module):
    def __init__(self, num_classes=5):
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

def get_resnet50(num_classes=5):
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def get_convnext_tiny(num_classes=5):
    model = models.convnext_tiny(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    return model

def get_swin_tiny(num_classes=5):
    model = models.swin_t(weights=None)
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)
    return model

# ---------------------------------------------------------
# CLASSE GRAD-CAM PARA CAPTURA DE GRADIENTES E ATIVAÇÕES
# ---------------------------------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Hooks para interceptar ativações e gradientes
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        gradients = self.gradients.data.cpu().numpy()[0]
        activations = self.activations.data.cpu().numpy()[0]

        # Trata diferença entre formato CNN (C, H, W) e Swin (H, W, C)
        if len(activations.shape) == 3 and activations.shape[0] not in [7, 14, 28, 56]:
            # Formato CNN: (C, H, W)
            weights = np.mean(gradients, axis=(1, 2))
            cam = np.zeros(activations.shape[1:], dtype=np.float32)
            for i, w in enumerate(weights):
                cam += w * activations[i, :, :]
        else:
            # Formato Swin-Transformer: (H, W, C)
            weights = np.mean(gradients, axis=(0, 1))
            cam = np.zeros(activations.shape[:2], dtype=np.float32)
            for i, w in enumerate(weights):
                cam += w * activations[:, :, i]

        cam = np.maximum(cam, 0)
        if np.max(cam) > 0:
            cam = cam / np.max(cam)

        return cam, class_idx

# ---------------------------------------------------------
# ANÁLISE QUANTITATIVA: CENTRO DO OBJETO VS. BORDA SINTÉTICA
# ---------------------------------------------------------
def analyze_heatmap_activation_ratio(heatmap, img_shape=(224, 224)):
    """
    Calcula a Razão de Ativação do Objeto (Ra):
    - Região Interna (Miolos): Área central do objeto (Erosão)
    - Região de Borda Sintética: Anel externo periférico do objeto (Dilatação - Erosão)

    Retorna:
    - ratio > 1.0 -> Ativação focada no INTERIOR do objeto (Sem Shortcut)
    - ratio < 1.0 -> Ativação concentrada nas BORDAS de recorte (Risco de Shortcut)
    """
    h_resized = cv2.resize(heatmap, img_shape)

    # Limiar para detectar área de interesse
    threshold_map = (h_resized > 0.3).astype(np.uint8)

    # Máscara do Anel de Borda vs Centro
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    dilated = cv2.dilate(threshold_map, kernel, iterations=1)
    eroded = cv2.erode(threshold_map, kernel, iterations=1)

    border_ring_mask = (dilated - eroded) > 0
    core_mask = eroded > 0

    core_energy = np.mean(h_resized[core_mask]) if np.sum(core_mask) > 0 else 0
    border_energy = np.mean(h_resized[border_ring_mask]) if np.sum(border_ring_mask) > 0 else 0

    ratio = (core_energy / (border_energy + 1e-6))
    return ratio, core_energy, border_energy

# ---------------------------------------------------------
# GERADOR DE MAPAS DE CALOR COMPLETO
# ---------------------------------------------------------
def run_xai_gradcam(model_path, model_type, image_path, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 5

    # Instancia o modelo correto
    if "convnext" in model_type:
        model = get_convnext_tiny(num_classes)
        target_layer = model.features[-1]
    elif "resnet" in model_type:
        model = get_resnet50(num_classes)
        target_layer = model.layer4[-1]
    elif "cnn" in model_type:
        model = CustomCNN(num_classes)
        target_layer = model.features[12]
    elif "swin" in model_type:
        model = get_swin_tiny(num_classes)
        target_layer = model.norm
    else:
        raise ValueError(f"Tipo de modelo não suportado: {model_type}")

    # Carrega pesos salvos
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    # Prepara a imagem de entrada
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    orig_img = Image.open(image_path).convert("RGB")
    orig_np = np.array(orig_img.resize((224, 224)))
    input_tensor = transform(orig_img).unsqueeze(0).to(device)

    # Executa Grad-CAM
    gradcam = GradCAM(model, target_layer)
    heatmap, pred_idx = gradcam.generate_heatmap(input_tensor)

    # Redimensiona o mapa de calor para 224x224
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

    # Sobreposição transparente (Overlay)
    overlay = cv2.addWeighted(orig_np, 0.6, heatmap_color, 0.4, 0)

    # Análise quantitativa de Borda vs Centro
    ratio, core_e, border_e = analyze_heatmap_activation_ratio(heatmap)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    classes = ['PET', 'PE_HD', 'PP', 'PS', 'Other']
    pred_class = classes[pred_idx] if pred_idx < len(classes) else f"Classe_{pred_idx}"

    status_shortcut = "[OK] FOCO NO OBJETO (Sem Shortcut)" if ratio >= 1.0 else "[ALERTA] FOCO NA BORDA (Risco de Shortcut)"

    print(f"\n[XAI GRAD-CAM] Imagem: {os.path.basename(image_path)}")
    print(f"  - Classe Predita: {pred_class}")
    print(f"  - Energia no Miolo do Objeto: {core_e:.3f}")
    print(f"  - Energia na Borda Sintetica: {border_e:.3f}")
    print(f"  - Razao Miolo/Borda: {ratio:.3f}")
    print(f"  - Avaliacao: {status_shortcut}")
    print(f"  - Mapa salvo em: {output_path}")

    return pred_class, ratio, output_path

def generate_multi_model_grid_panel(image_path, models_dir, output_grid_path):
    """
    Gera automaticamente um painel visual em grade comparativa (Side-by-Side)
    mostrando a mesma imagem sintética sob todos os modelos (ConvNeXt, ResNet, Swin-T, Custom CNN).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    classes = ['Other', 'PET', 'PE_HD', 'PP', 'PS']

    orig_pil = Image.open(image_path).convert("RGB")
    orig_np = np.array(orig_pil.resize((224, 224)))
    input_tensor = transform(orig_pil).unsqueeze(0).to(device)

    models_to_test = [
        ("modelo_convnext_looo.pth", "convnext", "ConvNeXt-Tiny"),
        ("modelo_resnet_looo.pth", "resnet", "ResNet-50"),
        ("modelo_swin_looo.pth", "swin", "Swin-Transformer"),
        ("modelo_cnn_looo.pth", "cnn", "Custom CNN")
    ]

    # Bloco 1: Imagem Original Sintética
    blocks = []
    blk_orig = np.zeros((224 + 32, 224, 3), dtype=np.uint8)
    lbl_orig = np.zeros((32, 224, 3), dtype=np.uint8)
    cv2.putText(lbl_orig, "Imagem Sintetica Original", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    blk_orig[:32, :] = lbl_orig
    blk_orig[32:, :] = orig_np
    blocks.append(blk_orig)

    for pth_name, arch_type, display_name in models_to_test:
        pth_path = os.path.join(models_dir, pth_name)
        if not os.path.exists(pth_path):
            continue

        if "convnext" in arch_type:
            model = get_convnext_tiny(5)
            target_layer = model.features[-1]
        elif "resnet" in arch_type:
            model = get_resnet50(5)
            target_layer = model.layer4[-1]
        elif "cnn" in arch_type:
            model = CustomCNN(5)
            target_layer = model.features[12]
        elif "swin" in arch_type:
            model = get_swin_tiny(5)
            target_layer = model.norm

        model.load_state_dict(torch.load(pth_path, map_location=device))
        model.to(device)
        model.eval()

        gradcam = GradCAM(model, target_layer)
        heatmap, pred_idx = gradcam.generate_heatmap(input_tensor)
        pred_class = classes[pred_idx] if pred_idx < len(classes) else f"Classe_{pred_idx}"

        heatmap_resized = cv2.resize(heatmap, (224, 224))
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(orig_np, 0.6, heatmap_color, 0.4, 0)

        blk = np.zeros((224 + 32, 224, 3), dtype=np.uint8)
        lbl = np.zeros((32, 224, 3), dtype=np.uint8)
        cv2.putText(lbl, f"{display_name}", (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(lbl, f"Pred: {pred_class}", (5, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)
        blk[:32, :] = lbl
        blk[32:, :] = overlay
        blocks.append(blk)

    if blocks:
        # Monta a grade lado a lado
        row1 = np.hstack(blocks[:3])
        row2 = np.hstack(blocks[3:])
        # Completa row2 se necessário
        if len(blocks[3:]) < 3:
            dummy_fill = np.zeros((224 + 32, 224 * (3 - len(blocks[3:])), 3), dtype=np.uint8)
            row2 = np.hstack([row2, dummy_fill])

        full_grid = np.vstack([row1, row2])

        os.makedirs(os.path.dirname(output_grid_path), exist_ok=True)
        cv2.imwrite(output_grid_path, cv2.cvtColor(full_grid, cv2.COLOR_RGB2BGR))
        print(f"  - [GRADE COMPARATIVA] Painel com Swin-T, ConvNeXt, ResNet e CNN salvo em: {output_grid_path}")
        return output_grid_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Ferramenta de Auditoria XAI Grad-CAM e Detecção de Shortcut Learning")
    parser.add_argument("--model", type=str, default=r"C:\Users\Vinicius\Desktop\MestradoCodeAnti\modelos0408\modelo_convnext_looo.pth", help="Caminho do arquivo .pth do modelo")
    parser.add_argument("--model_type", type=str, default="convnext", choices=["convnext", "resnet", "cnn", "swin"], help="Tipo da arquitetura")
    parser.add_argument("--img_dir", type=str, required=True, help="Diretório ou caminho da imagem de teste")
    parser.add_argument("--output_dir", type=str, default="./Resultados_Modelos/v5_benchmark_expanded_results_v2/xai_heatmaps", help="Diretório para salvar os mapas de calor")
    parser.add_argument("--models_dir", type=str, default=r"C:\Users\Vinicius\Desktop\MestradoCodeAnti\modelos0408", help="Pasta com todos os modelos salvos")
    parser.add_argument("--limit", type=int, default=5, help="Quantidade de imagens a analisar")

    args = parser.parse_args()

    if os.path.isfile(args.img_dir):
        imgs = [args.img_dir]
    elif os.path.exists(args.img_dir):
        imgs = [os.path.join(args.img_dir, f) for f in os.listdir(args.img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:args.limit]
    else:
        imgs = []

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n=======================================================")
    print(f"AUDITORIA XAI GRAD-CAM: {args.model_type.upper()} ({len(imgs)} imagens)")
    print(f"=======================================================")

    for i, img_path in enumerate(imgs):
        out_name = f"{args.model_type}_gradcam_{i+1}_{os.path.basename(img_path)}"
        out_path = os.path.join(args.output_dir, out_name)
        pred_class, ratio, saved_p = run_xai_gradcam(args.model, args.model_type, img_path, out_path)

        # Gera AUTOMATICAMENTE o painel em grade comparativo com TODOS os modelos (incluindo Swin-T)
        grid_out_name = f"GRADE_COMPARATIVA_{i+1}_{os.path.basename(img_path)}"
        grid_out_path = os.path.join(args.output_dir, "grades_comparativas", grid_out_name)
        generate_multi_model_grid_panel(img_path, args.models_dir, grid_out_path)

    print(f"=======================================================\n")


