import os
import cv2
import time
import argparse
import collections
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# --- Arquiteturas dos Modelos para carregar os pesos (.pth) ---

class CustomCNN_V2(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN_V2, self).__init__()
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
        return self.classifier(self.features(x))

def build_resnet_v2(num_classes):
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def build_vit_model(num_classes):
    model = models.vit_b_16(weights=None)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    return model

def build_convnext_v2(num_classes):
    model = models.convnext_tiny(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    return model

def build_swin_v2(num_classes):
    model = models.swin_t(weights=None)
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)
    return model

# --- Configuração de Dispositivo ---
device = torch.device("cpu")
try:
    import torch_directml
    if torch_directml.is_available():
        device = torch_directml.device()
except ImportError:
    pass

# --- Função do Atuador Simulador ---
def trigger_actuator(class_name, confidence):
    print(f"\n>>> [GATILHO] ATUADOR DISPARADO! Classe: {class_name} | Confiança: {confidence:.2f} | Horário: {time.strftime('%H:%M:%S')}")

def main():
    parser = argparse.ArgumentParser(description="Simulador de Classificação de Esteira em Tempo Real")
    parser.add_argument("--source", type=str, default="0", help="Caminho do arquivo de vídeo ou '0' para webcam")
    parser.add_argument("--model_type", type=str, default="resnet", choices=["resnet", "cnn", "vit", "convnext", "swin"], help="Tipo de modelo ('resnet', 'cnn', 'vit', 'convnext' ou 'swin')")
    parser.add_argument("--weights", type=str, default="modelo_transfer_v2.pth", help="Caminho do arquivo de pesos (.pth)")
    parser.add_argument("--threshold", type=float, default=0.85, help="Limiar de confiança para disparo (0.0 a 1.0)")
    parser.add_argument("--buffer_size", type=int, default=5, help="Tamanho do buffer para média móvel temporal")
    parser.add_argument("--cooldown", type=float, default=1.5, help="Tempo de bloqueio pós-disparo (segundos)")
    
    args = parser.parse_args()
    
    class_names = ['Other', 'PET', 'PE_HD', 'PP', 'PS']
    num_classes = len(class_names)
    
    # 1. Carregar Modelo
    print(f"Carregando modelo {args.model_type} com pesos {args.weights}...")
    if args.model_type == "resnet":
        model = build_resnet_v2(num_classes)
    elif args.model_type == "vit":
        model = build_vit_model(num_classes)
    elif args.model_type == "convnext":
        model = build_convnext_v2(num_classes)
    elif args.model_type == "swin":
        model = build_swin_v2(num_classes)
    else:
        model = CustomCNN_V2(num_classes)
        
    if not os.path.exists(args.weights):
        print(f"ERRO: Arquivo de pesos {args.weights} não encontrado no diretório!")
        return
        
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)
    model.eval()
    print("Modelo carregado com sucesso!")
    
    # 2. Configurar Captura de Vídeo
    source = args.source
    if source.isdigit():
        source = int(source)
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"ERRO: Não foi possível abrir o feed de vídeo: {args.source}")
        return
        
    # 3. Transformações de Preprocessamento
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 4. Estado da Fila de Suavização e Cooldown
    probs_buffer = collections.deque(maxlen=args.buffer_size)
    last_trigger_time = 0
    
    print("\nIniciando inferência em tempo real...")
    print("Pressione 'q' na janela de vídeo para sair.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Fim do vídeo ou sinal da câmera perdido.")
            break
            
        # Manter cópia para renderização
        display_frame = frame.copy()
        
        # Preprocessar Frame
        # Converter BGR do OpenCV para RGB do PIL
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        input_tensor = val_transform(pil_img).unsqueeze(0).to(device)
        
        # Inferência
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            
        # Adicionar histórico ao buffer
        probs_buffer.append(probabilities)
        
        # Calcular média móvel
        avg_probs = np.mean(probs_buffer, axis=0)
        pred_class_idx = np.argmax(avg_probs)
        pred_prob = avg_probs[pred_class_idx]
        pred_class_name = class_names[pred_class_idx]
        
        # Lógica de Disparo e Cooldown
        current_time = time.time()
        cooldown_remaining = max(0.0, args.cooldown - (current_time - last_trigger_time))
        cooldown_active = cooldown_remaining > 0
        
        if pred_class_name != 'Other' and pred_prob >= args.threshold:
            if not cooldown_active:
                trigger_actuator(pred_class_name, pred_prob)
                last_trigger_time = current_time
                cooldown_active = True
                
        # --- Interface Visual Overlay (GUI OpenCV) ---
        # Desenhar painel escuro de background
        cv2.rectangle(display_frame, (10, 10), (320, 200), (0, 0, 0), -1)
        
        # Exibir classe ativa e probabilidade
        color = (0, 255, 0) if pred_class_name != 'Other' else (0, 165, 255)
        cv2.putText(display_frame, f"Classe: {pred_class_name}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(display_frame, f"Conf: {pred_prob*100:.1f}%", (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Exibir barras de probabilidade de todas as classes
        for i, (name, prob) in enumerate(zip(class_names, avg_probs)):
            y_pos = 100 + i * 18
            # Texto da classe
            cv2.putText(display_frame, f"{name}:", (20, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            # Barra gráfica
            bar_width = int(prob * 100)
            cv2.rectangle(display_frame, (90, y_pos - 8), (90 + bar_width, y_pos + 2), (255, 0, 0), -1)
            cv2.putText(display_frame, f"{prob*100:.0f}%", (195, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
        # Exibir status do cooldown do atuador
        if cooldown_active:
            cv2.putText(display_frame, f"COOLDOWN: {cooldown_remaining:.1f}s", (20, 220), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(display_frame, "PRONTO PARA DISPARO", (20, 220), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
        # Mostrar imagem
        cv2.imshow("Classificador da Esteira (Simulação)", display_frame)
        
        # Tecla de Saída 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print("Simulador encerrado.")

if __name__ == '__main__':
    main()
