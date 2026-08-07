import os
import sys
import cv2
import glob
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

sys.path.append(r"C:\Users\Vinicius\Desktop\MestradoCodeAnti\Codigos\PlasticRecognition")
from pipeline.xai_gradcam_analysis import GradCAM, CustomCNN, get_resnet50, get_convnext_tiny, get_swin_tiny

def find_available_models(base_dir=r"C:\Users\Vinicius\Desktop\MestradoCodeAnti"):
    """Varre o projeto em busca de todos os modelos .pth salvos."""
    pth_files = []
    for root, _, files in os.walk(base_dir):
        if "venv" in root or ".git" in root or ".gemini" in root:
            continue
        for f in sorted(files):
            if f.endswith(".pth") and not f.startswith("."):
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, base_dir)
                pth_files.append((f, full_path, rel_path))
    return pth_files

def infer_architecture_type(model_filename):
    """Infere a arquitetura baseada no nome do arquivo .pth."""
    name = model_filename.lower()
    if "convnext" in name:
        return "convnext"
    elif "resnet" in name:
        return "resnet"
    elif "swin" in name:
        return "swin"
    elif "cnn" in name:
        return "cnn"
    elif "transfer" in name:
        return "resnet"
    return "convnext"

def load_model_and_target_layer(pth_path, arch_type, num_classes=5, device="cpu"):
    """Instancia a arquitetura e carrega os pesos salvos."""
    if arch_type == "convnext":
        model = get_convnext_tiny(num_classes)
        target_layer = model.features[-1]
    elif arch_type == "resnet":
        model = get_resnet50(num_classes)
        target_layer = model.layer4[-1]
    elif arch_type == "cnn":
        model = CustomCNN(num_classes)
        target_layer = model.features[12]
    elif arch_type == "swin":
        model = get_swin_tiny(num_classes)
        target_layer = model.norm
    else:
        model = get_convnext_tiny(num_classes)
        target_layer = model.features[-1]

    state_dict = torch.load(pth_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model, target_layer

def generate_comparative_grid_for_image(img_path, selected_models, output_grid_path, device="cpu"):
    """
    Gera a grade comparativa de 1 imagem passando por todos os modelos selecionados.
    selected_models: Lista de tuplas (nome_display, pth_path, arch_type)
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    classes = ['Other', 'PET', 'PE_HD', 'PP', 'PS']

    orig_pil = Image.open(img_path).convert("RGB")
    orig_np = np.array(orig_pil.resize((224, 224)))
    input_tensor = transform(orig_pil).unsqueeze(0).to(device)

    panels = []

    # Bloco 1: Imagem Original
    blk_orig = np.zeros((224 + 32, 224, 3), dtype=np.uint8)
    lbl_orig = np.zeros((32, 224, 3), dtype=np.uint8)
    cv2.putText(lbl_orig, "Imagem Original", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    blk_orig[:32, :] = lbl_orig
    blk_orig[32:, :] = orig_np
    panels.append(blk_orig)

    # Gera o mapa de calor para cada modelo selecionado
    for display_name, pth_path, arch_type in selected_models:
        try:
            model, target_layer = load_model_and_target_layer(pth_path, arch_type, num_classes=5, device=device)
            gradcam = GradCAM(model, target_layer)
            heatmap, pred_idx = gradcam.generate_heatmap(input_tensor)
            pred_class = classes[pred_idx] if pred_idx < len(classes) else f"Classe_{pred_idx}"

            heatmap_resized = cv2.resize(heatmap, (224, 224))
            heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(orig_np, 0.6, heatmap_color, 0.4, 0)

            block = np.zeros((224 + 32, 224, 3), dtype=np.uint8)
            lbl = np.zeros((32, 224, 3), dtype=np.uint8)
            cv2.putText(lbl, f"{display_name[:20]}", (5, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(lbl, f"Pred: {pred_class}", (5, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)
            block[:32, :] = lbl
            block[32:, :] = overlay
            panels.append(block)
        except Exception as e:
            print(f"[AVISO] Erro ao processar modelo {display_name}: {e}")

    if len(panels) > 1:
        # Organiza a grade dinamicamente (máximo 4 colunas por linha)
        cols_per_row = min(len(panels), 4)
        rows = []
        for idx in range(0, len(panels), cols_per_row):
            row_panels = panels[idx : idx + cols_per_row]
            if len(row_panels) < cols_per_row:
                dummy_fill = np.zeros((224 + 32, 224 * (cols_per_row - len(row_panels)), 3), dtype=np.uint8)
                row_panels.append(dummy_fill)
            rows.append(np.hstack(row_panels))

        full_grid = np.vstack(rows)
        os.makedirs(os.path.dirname(output_grid_path), exist_ok=True)
        cv2.imwrite(output_grid_path, cv2.cvtColor(full_grid, cv2.COLOR_RGB2BGR))
        print(f"  [OK] Grade criada ({len(selected_models)} modelos) -> {output_grid_path}")
        return output_grid_path
    return None

def main():
    print("==========================================================================")
    print("      PROGRAMA DE AUDITORIA E GRADE COMPARATIVA INTERATIVA XAI")
    print("==========================================================================")

    # 1. Encontra todos os modelos .pth do projeto
    all_pths = find_available_models()
    if not all_pths:
        print("[ERRO] Nenhum arquivo de modelo .pth encontrado.")
        return

    print("\n--- PASSO 1: MODELOS DISPONÍVEIS ENCONTRADOS ---")
    for idx, (filename, full_p, rel_p) in enumerate(all_pths):
        print(f"  [{idx + 1:02d}] {filename}  (Caminho: {rel_p})")

    user_model_idx = input("\nDigite os NÚMEROS dos modelos desejados separados por vírgula (ex: 1,3,5) ou 'ALL' para todos os de modelos0408: ").strip()

    selected_models = []
    if user_model_idx.upper() == "ALL" or not user_model_idx:
        # Seleciona os modelos principais da pasta modelos0408
        for filename, full_p, rel_p in all_pths:
            if "modelos0408" in rel_p and "pth" in filename:
                arch_t = infer_architecture_type(filename)
                display_n = filename.replace(".pth", "")
                selected_models.append((display_n, full_p, arch_t))
    else:
        indices = [int(x.strip()) - 1 for x in user_model_idx.split(",") if x.strip().isdigit()]
        for i in indices:
            if 0 <= i < len(all_pths):
                filename, full_p, _ = all_pths[i]
                arch_t = infer_architecture_type(filename)
                display_n = filename.replace(".pth", "")
                selected_models.append((display_n, full_p, arch_t))

    print(f"\n[OK] {len(selected_models)} modelos selecionados para a grade comparativa:")
    for name, pth, arch in selected_models:
        print(f"  - {name} ({arch.upper()})")

    # 2. Seleção de Imagens
    print("\n--- PASSO 2: SELEÇÃO DE IMAGENS ---")
    img_input = input("Cole o CAMINHO de uma imagem (ex: C:\\foto.jpg) ou de uma PASTA de imagens: ").strip('"\' ')

    images_to_process = []
    if os.path.isfile(img_input):
        images_to_process.append(img_input)
    elif os.path.isdir(img_input):
        valid_exts = ('.jpg', '.jpeg', '.png')
        images_to_process = [os.path.join(img_input, f) for f in sorted(os.listdir(img_input)) if f.lower().endswith(valid_exts)]
    else:
        print(f"[ERRO] Caminho inválido: {img_input}")
        return

    print(f"\n[OK] {len(images_to_process)} imagens encontradas para testar.")

    # 3. Geração das Grades Comparativas
    out_dir = r"C:\Users\Vinicius\Desktop\MestradoCodeAnti\Resultados_Modelos\v5_benchmark_expanded_results_v2\xai_interactive_grids"
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n==========================================================================")
    print("GERANDO GRADES COMPARATIVAS DE MAPA DE CALOR...")
    print("==========================================================================")

    for idx, img_p in enumerate(images_to_process):
        img_name = os.path.basename(img_p)
        grid_file_name = f"GRADE_COMPARATIVA_{idx + 1:02d}_{img_name}"
        out_grid_p = os.path.join(out_dir, grid_file_name)
        generate_comparative_grid_for_image(img_p, selected_models, out_grid_p, device=device)

    print(f"\n==========================================================================")
    print(f"[CONCLUÍDO] Todas as grades comparativas foram geradas com sucesso!")
    print(f"📂 Pasta de Destino: {out_dir}")
    print("==========================================================================\n")

if __name__ == '__main__':
    main()
