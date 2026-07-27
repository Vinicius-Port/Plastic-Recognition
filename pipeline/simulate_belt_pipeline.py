import os
import cv2
import time
import json
import argparse
import collections
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

from train_pipeline import build_model, get_transforms

# Configuração de Dispositivo
device = torch.device("cpu")
try:
    import torch_directml
    if torch_directml.is_available():
        device = torch_directml.device()
except ImportError:
    pass

if device.type == "cpu" and torch.cuda.is_available():
    device = torch.device("cuda")

CLASS_NAMES = ['Other', 'PET', 'PE_HD', 'PP', 'PS']

# Limiares de Confiança por Classe para o Atuador de Descarte
CLASS_THRESHOLDS = {
    'PET': 0.75,
    'PE_HD': 0.75,
    'PP': 0.70,
    'PS': 0.75,
    'Other': 0.90
}

def load_ground_truth(gt_path="belt_ground_truth.json"):
    if os.path.exists(gt_path):
        with open(gt_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def get_current_ground_truth(timestamp_sec, gt_data):
    if not gt_data or "events" not in gt_data:
        return None, None
    for event in gt_data["events"]:
        if event["start_sec"] <= timestamp_sec <= event["end_sec"]:
            return event.get("ground_truth_class"), event.get("object_id")
    return "Other", None

POSSIBLE_MODEL_PATHS = [
    "outputs/modelo_resnet_looo/model.pth",
    "outputs/modelo_convnext_looo/model.pth",
    "modelo_transfer_v2.pth",
    "modelo_resnet_v2.pth",
    "modelo_cnn_v2.pth",
    "modelo_convnext_v2.pth",
    "modelo_swin_v2.pth"
]

def resolve_model_path(requested_path=None):
    if requested_path and os.path.exists(requested_path):
        return requested_path
    for cand in POSSIBLE_MODEL_PATHS:
        if os.path.exists(cand):
            print(f"[MODELO] Usando pesos encontrados em: {cand}")
            return cand
    if requested_path:
        return requested_path
    raise FileNotFoundError("Nenhum arquivo de pesos (.pth) foi encontrado!")

POSSIBLE_VIDEO_PATHS = [
    "data/simulation_belt.mp4",
    "simulation_belt.mp4",
    "../data/simulation_belt.mp4"
]

POSSIBLE_GT_PATHS = [
    "data/belt_ground_truth.json",
    "belt_ground_truth.json",
    "../data/belt_ground_truth.json"
]

def resolve_video_path(requested_path=None):
    if requested_path and os.path.exists(requested_path):
        return requested_path
    for cand in POSSIBLE_VIDEO_PATHS:
        if os.path.exists(cand):
            return cand
    return requested_path if requested_path else "data/simulation_belt.mp4"

def resolve_gt_path(requested_path=None):
    if requested_path and os.path.exists(requested_path):
        return requested_path
    for cand in POSSIBLE_GT_PATHS:
        if os.path.exists(cand):
            return cand
    return requested_path if requested_path else "data/belt_ground_truth.json"

def run_belt_simulation(model_path, arch, video_path="data/simulation_belt.mp4", gt_path="data/belt_ground_truth.json", output_dir=None, show_window=True, cooldown_sec=1.5):
    model_path = resolve_model_path(model_path)
    video_path = resolve_video_path(video_path)
    gt_path = resolve_gt_path(gt_path)
    print("\n=======================================================")
    print(f"SIMULAÇÃO NA ESTEIRA EM TEMPO REAL: {os.path.basename(model_path)}")
    print(f"Vídeo: {video_path} | Gabarito: {gt_path}")
    print("=======================================================")

    if not os.path.exists(video_path):
        print(f"[ERRO] Vídeo da esteira não encontrado: {video_path}")
        return None

    # Determine Output Directory
    if not output_dir:
        output_dir = os.path.dirname(model_path) if os.path.dirname(model_path) else "./outputs/simulation_result"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Carregar Split Info (para verificar se o objeto é inédito no LOOO)
    split_info_path = os.path.join(output_dir, "split_info.json")
    val_objects = set()
    if os.path.exists(split_info_path):
        with open(split_info_path, "r", encoding="utf-8") as f:
            split_info = json.load(f)
            val_objects = set(split_info.get("val_objects", []))

    # 2. Carregar Modelo
    model = build_model(arch, num_classes=len(CLASS_NAMES)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    _, val_transform = get_transforms()
    gt_data = load_ground_truth(gt_path)

    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0: video_fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Métricas de Simulação
    frame_idx = 0
    predictions_log = []
    correct_evaluations = 0
    total_evaluations = 0
    actuator_triggers = 0
    last_trigger_time = 0.0

    prediction_buffer = collections.deque(maxlen=5)

    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        current_sec = frame_idx / video_fps

        # Detection of Object Region (Simple Color/Contour Detection for Bounding Box)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bbox = None
        max_area = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 3000:  # Objeto detectado na esteira
                x, y, w, h = cv2.boundingRect(cnt)
                if area > max_area:
                    max_area = area
                    bbox = (x, y, w, h)

        # Ground Truth lookup
        gt_class, obj_id = get_current_ground_truth(current_sec, gt_data)

        # Inference on Frame / ROI
        if bbox:
            x, y, w, h = bbox
            roi = frame[y:y+h, x:x+w]
            img_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        else:
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        tensor_img = val_transform(img_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(tensor_img)
            probs = torch.softmax(outputs, dim=1)[0]
            conf, pred_idx = torch.max(probs, 0)
            pred_class = CLASS_NAMES[pred_idx.item()]
            confidence = conf.item()

        prediction_buffer.append(pred_class)
        # Smoothing
        smoothed_pred = collections.Counter(prediction_buffer).most_common(1)[0][0]

        # Actuator Trigger Logic
        required_thresh = CLASS_THRESHOLDS.get(smoothed_pred, 0.75)
        actuator_active = False
        if smoothed_pred != "Other" and confidence >= required_thresh:
            now = time.time()
            if (now - last_trigger_time) > cooldown_sec:
                actuator_active = True
                actuator_triggers += 1
                last_trigger_time = now

        # Evaluate Prediction vs Ground Truth
        if gt_class:
            total_evaluations += 1
            is_correct = (smoothed_pred == gt_class)
            if is_correct:
                correct_evaluations += 1
        else:
            is_correct = None

        current_acc = (correct_evaluations / total_evaluations * 100) if total_evaluations > 0 else 0.0

        # Check if Object is in Validation Set (LOOO)
        is_val_object = (obj_id and obj_id in val_objects)

        # Render Overlay UI
        if show_window:
            overlay = frame.copy()

            # Draw Bounding Box
            if bbox:
                x, y, w, h = bbox
                box_color = (0, 255, 0) if is_correct else (0, 0, 255)
                cv2.rectangle(overlay, (x, y), (x + w, y + h), box_color, 2)
                cv2.putText(overlay, f"{smoothed_pred} ({confidence*100:.1f}%)", (x, max(20, y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

            # Top Info Bar
            cv2.rectangle(overlay, (0, 0), (640, 75), (30, 30, 30), -1)

            # Left Info: Pred vs GT
            cv2.putText(overlay, f"Predicao: {smoothed_pred} ({confidence*100:.1f}%)", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            gt_text = f"Gabarito: {gt_class if gt_class else 'N/A'}"
            cv2.putText(overlay, gt_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # Right Info: Accuracy & Actuator
            cv2.putText(overlay, f"Acuracia Esteira: {current_acc:.1f}%", (360, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if actuator_active or (time.time() - last_trigger_time < 0.5):
                cv2.putText(overlay, "[⚡ ATUADOR DISPARADO]", (360, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                cv2.putText(overlay, "Atuador: Standby", (360, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

            # Badge LOOO Validation Object
            if is_val_object:
                cv2.rectangle(overlay, (10, 440), (320, 470), (0, 100, 0), -1)
                cv2.putText(overlay, "🟢 [OBJETO INEDITO - VALIDAÇÃO LOOO]", (15, 462),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            cv2.imshow("Simulador de Esteira de Triagem - AI/ML Pipeline", overlay)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    elapsed_time = time.time() - start_time
    calc_fps = total_frames / elapsed_time if elapsed_time > 0 else 0.0

    # Save Simulation Report
    sim_report = {
        "model_path": model_path,
        "architecture": arch,
        "video_processed": video_path,
        "total_frames": total_frames,
        "simulation_fps": calc_fps,
        "total_evaluations": total_evaluations,
        "correct_evaluations": correct_evaluations,
        "treadmill_accuracy_percent": current_acc,
        "actuator_total_triggers": actuator_triggers
    }

    with open(os.path.join(output_dir, "belt_simulation_report.json"), "w", encoding="utf-8") as f:
        json.dump(sim_report, f, indent=2, ensure_ascii=False)

    print("\n=======================================================")
    print("RESULTADO FINAL DA SIMULAÇÃO NA ESTEIRA")
    print("=======================================================")
    print(f"Model: {os.path.basename(model_path)}")
    print(f"Frames Processados: {total_frames} ({calc_fps:.1f} FPS)")
    print(f"Acurácia Acumulada na Esteira: {current_acc:.2f}% ({correct_evaluations}/{total_evaluations})")
    print(f"Disparos do Atuador de Descarte: {actuator_triggers} vezes")
    print(f"Relatório Salvo: {os.path.join(output_dir, 'belt_simulation_report.json')}")
    print("=======================================================\n")

    return sim_report

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simulador de Esteira em Tempo Real com Gabarito")
    parser.add_argument("--model_path", type=str, default="outputs/modelo_resnet_looo/model.pth", help="Caminho dos pesos do modelo (.pth)")
    parser.add_argument("--arch", type=str, default="resnet", choices=["cnn", "resnet", "convnext", "swin"], help="Arquitetura do modelo")
    parser.add_argument("--video", type=str, default="simulation_belt.mp4", help="Vídeo da esteira")
    parser.add_argument("--gt", type=str, default="belt_ground_truth.json", help="Arquivo de Gabarito (Ground Truth)")
    parser.add_argument("--headless", action="store_true", help="Executar em modo headless (sem janela gráfica)")

    args = parser.parse_args()
    run_belt_simulation(args.model_path, args.arch, video_path=args.video, gt_path=args.gt, show_window=not args.headless)
