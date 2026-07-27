import os
import json
import torch
import numpy as np
from train_pipeline import CustomCNN, build_model

def compile_leaderboard(outputs_dir="outputs"):
    """
    Analisa todos os modelos em outputs/ e gera uma tabela classificatória (Leaderboard).
    """
    if not os.path.exists(outputs_dir):
        print(f"[AVISO] Pasta '{outputs_dir}' ainda não existe.")
        return []

    models_data = []
    for entry in os.listdir(outputs_dir):
        model_folder = os.path.join(outputs_dir, entry)
        metrics_file = os.path.join(model_folder, "metrics.json")
        split_file = os.path.join(model_folder, "split_info.json")

        if os.path.isdir(model_folder) and os.path.exists(metrics_file):
            with open(metrics_file, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            
            split_info = {}
            if os.path.exists(split_file):
                with open(split_file, "r", encoding="utf-8") as f:
                    split_info = json.load(f)

            models_data.append({
                "folder": entry,
                "model_name": metrics.get("model_name", entry),
                "accuracy": metrics.get("accuracy", 0.0),
                "weighted_f1": metrics.get("weighted_f1", 0.0),
                "strategy": split_info.get("strategy", "unknown"),
                "epochs": len(metrics.get("history", {}).get("accuracy", []))
            })

    # Sort by accuracy descending
    models_data.sort(key=lambda x: x["accuracy"], reverse=True)

    print("\n=========================================================================")
    print("🏆 LEADERBOARD DOS MODELOS TREINADOS (RANKING DE DESEMPENHO)")
    print("=========================================================================")
    print(f"{'Posição':<8} {'Modelo':<25} {'Estratégia':<12} {'Épocas':<8} {'Acurácia':<10} {'F1-Weighted':<10}")
    print("-" * 75)
    for idx, item in enumerate(models_data, 1):
        print(f"{idx:<8} {item['model_name']:<25} {item['strategy']:<12} {item['epochs']:<8} {item['accuracy']*100:>6.2f}%    {item['weighted_f1']*100:>6.2f}%")
    print("=========================================================================\n")

    # Salva Leaderboard JSON
    with open(os.path.join(outputs_dir, "leaderboard.json"), "w", encoding="utf-8") as f:
        json.dump(models_data, f, indent=2, ensure_ascii=False)

    return models_data

def export_model_onnx(model_path, arch, num_classes=5, output_onnx_path=None):
    """
    Exporta o modelo PyTorch (.pth) para o formato ONNX para inferência ultra-rápida.
    """
    if not os.path.exists(model_path):
        print(f"[ERRO] Arquivo de pesos não encontrado: {model_path}")
        return None

    if not output_onnx_path:
        output_onnx_path = model_path.replace(".pth", ".onnx")

    device = torch.device("cpu")
    model = build_model(arch, num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    try:
        torch.onnx.export(
            model, dummy_input, output_onnx_path,
            export_params=True, opset_version=12,
            do_constant_folding=True,
            input_names=['input'], output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print(f"[EXPORTAÇÃO ONNX] Modelo exportado com sucesso -> {output_onnx_path}")
        return output_onnx_path
    except Exception as e:
        print(f"[ERRO ONNX] Falha ao exportar ONNX: {e}")
        return None

if __name__ == '__main__':
    compile_leaderboard()
