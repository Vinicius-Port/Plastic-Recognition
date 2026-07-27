import os
import re
import cv2
import shutil
import numpy as np
from collections import defaultdict

DATA_DIR = r"C:\Users\Vinicius\Desktop\MestradoCodeAnti\ProjetoPlasticRecognition\Datasets\Dataset_Wadaba"
ARTIFACT_DIR = r"C:\Users\Vinicius\.gemini\antigravity\brain\85fe9ecc-198c-4540-a583-52f5b4f5eff1"
ATTACHED_IMG = os.path.join(ARTIFACT_DIR, "media__1784312228186.jpg")
LOCAL_TEXTURE = r"C:\Users\Vinicius\Desktop\MestradoCodeAnti\ProjetoPlasticRecognition\PlasticRecognitionLocal\esteira_textura.jpg"
OUTPUT_VIDEO = r"C:\Users\Vinicius\Desktop\MestradoCodeAnti\ProjetoPlasticRecognition\PlasticRecognitionLocal\simulation_belt.mp4"

# 1. Obter objetos de validação (semente 123)
def get_val_objects(data_dir):
    class_names = ['Other', 'PET', 'PE_HD', 'PP', 'PS']
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    code_to_class = {'01': 'PET', '02': 'PE_HD', '05': 'PP', '06': 'PS', '07': 'Other'}
    
    img_paths = []
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_paths.append(os.path.join(root, f))
                
    object_images = defaultdict(lambda: defaultdict(list))
    for path in img_paths:
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
        if match_obj:
            obj_id = match_obj.group(1)
        else:
            obj_id = filename
        object_images[class_name][obj_id].append(path)
        
    rng = np.random.default_rng(123)
    val_split = 0.2
    val_objects = {}
    
    for class_name in class_names:
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
        val_objects[class_name] = sorted(shuffled_objs[split_idx:])
        
    return val_objects, object_images

# EXTRAÇÃO DO OBJETO (Corte do fundo claro)
def extract_object_mask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mask = np.zeros_like(thresh)
    if contours:
        c = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(c)
        cv2.drawContours(mask, [hull], -1, 255, -1)
    else:
        mask = thresh
        
    return mask

def main():
    if not os.path.exists(DATA_DIR):
        print(f"ERRO: Diretório de dados {DATA_DIR} não encontrado.")
        return
        
    # Copiar a imagem anexada da esteira para o projeto local se existir
    if os.path.exists(ATTACHED_IMG):
        shutil.copy2(ATTACHED_IMG, LOCAL_TEXTURE)
        print(f"Textura da esteira copiada para: {LOCAL_TEXTURE}")

    val_objects, object_images = get_val_objects(DATA_DIR)
    
    # Selecionar 4 objetos de validação específicos (um de cada classe reciclável)
    selected_targets = [
        ('PET', val_objects['PET'][0] if val_objects['PET'] else None),
        ('PE_HD', val_objects['PE_HD'][0] if val_objects['PE_HD'] else None),
        ('PP', val_objects['PP'][0] if val_objects['PP'] else None),
        ('PS', val_objects['PS'][0] if val_objects['PS'] else None),
    ]
    
    # Configuração do VideoWriter
    width, height = 640, 480
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))
    
    # Carregar imagem de textura da esteira e cortar a marca d'água
    if os.path.exists(LOCAL_TEXTURE):
        bg_img = cv2.imread(LOCAL_TEXTURE)
        # O tamanho original é (681, 1024, 3)
        # Cortamos a região inferior direita (marca d'água) usando coordenadas seguras [0:550, 0:850]
        bg_cropped = bg_img[0:550, 0:850]
        bg_resized = cv2.resize(bg_cropped, (width, height))
    else:
        bg_resized = np.ones((height, width, 3), dtype=np.uint8) * 35
        
    print(f"Gerando vídeo de simulação com objetos segmentados: {OUTPUT_VIDEO}...")

    frame_count = 0
    
    # Loop de criação do vídeo
    for class_name, obj_id in selected_targets:
        if not obj_id:
            continue
            
        images_list = sorted(object_images[class_name][obj_id])
        if len(images_list) == 0:
            continue
            
        print(f"  -> Adicionando objeto {obj_id} da classe {class_name}...")
        
        # 1. Adicionar 60 frames de esteira vazia (2 segundos de intervalo)
        for _ in range(60):
            bg = bg_resized.copy()
            cv2.putText(bg, "GABARITO: Outro", (360, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            out.write(bg)
            frame_count += 1
            
        # 2. Deslizar o objeto verticalmente de cima para baixo
        obj_size = 180  # Tamanho do objeto
        start_y = -obj_size
        end_y = height
        
        # Carregar apenas a primeira imagem e extrair a máscara
        obj_img_path = images_list[0]
        obj_img = cv2.imread(obj_img_path)
        obj_img_resized = cv2.resize(obj_img, (obj_size, obj_size))
        
        # Obter a máscara de segmentação do objeto
        mask = extract_object_mask(obj_img_resized)
        
        for f in range(150):
            bg = bg_resized.copy()
            
            # Calcular Y atual
            t = f / 149.0
            y = int(start_y + t * (end_y - start_y))
            x = int((width - obj_size) / 2) # Centralizado horizontalmente
            
            # Recortar ROIs para composição usando a máscara
            x1, x2 = max(0, x), min(width, x + obj_size)
            y1, y2 = max(0, y), min(height, y + obj_size)
            
            ox1, ox2 = max(0, -x), min(obj_size, width - x)
            oy1, oy2 = max(0, -y), min(obj_size, height - y)
            
            if x1 < x2 and y1 < y2:
                # Obter fatias
                bg_roi = bg[y1:y2, x1:x2]
                obj_roi = obj_img_resized[oy1:oy2, ox1:ox2]
                mask_roi = mask[oy1:oy2, ox1:ox2]
                
                # Mesclar apenas os pixels do objeto (onde a máscara é branca)
                bg_roi[mask_roi == 255] = obj_roi[mask_roi == 255]
                
            # Adicionar texto do Gabarito na tela (Canto superior direito)
            cv2.putText(bg, f"GABARITO: {class_name}", (360, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            out.write(bg)
            frame_count += 1

    # Adicionar mais 60 frames de esteira vazia no final
    for _ in range(60):
        bg = bg_resized.copy()
        cv2.putText(bg, "GABARITO: Outro", (360, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        out.write(bg)
        frame_count += 1
        
    out.release()
    print(f"Vídeo de simulação gerado com sucesso! Total de frames: {frame_count} (~{frame_count/fps:.1f}s)")

if __name__ == '__main__':
    main()
