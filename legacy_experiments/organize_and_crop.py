import os
import re
import cv2
import shutil
import numpy as np

DATA_DIR = r"C:\Users\Vinicius\Desktop\MestradoCodeAnti\ProjetoPlasticRecognition\Datasets\Dataset_Wadaba"
OUTPUT_DIR = r"C:\Users\Vinicius\Desktop\MestradoCodeAnti\ProjetoPlasticRecognition\Datasets\Dataset_Wadaba_Cropped"

def extract_object_mask(img):
    # Converter para escala de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # CORREÇÃO: Otsu's thresholding para separar o fundo escuro (0) do objeto claro (255)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Limpeza morfológica
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Contorno externo para obter a borda do objeto e preencher reflexos internos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mask = np.zeros_like(thresh)
    if contours:
        c = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [c], -1, 255, -1)  # Desenha contorno preenchido
    else:
        mask = thresh
        
    return mask

def main():
    if not os.path.exists(DATA_DIR):
        print(f"ERRO: Diretório de dados {DATA_DIR} não encontrado.")
        return
        
    class_names = ['Other', 'PET', 'PE_HD', 'PP', 'PS']
    code_to_class = {
        '01': 'PET',
        '02': 'PE_HD',
        '05': 'PP',
        '06': 'PS',
        '07': 'Other'
    }
    
    print("Iniciando organização e recorte do dataset (Corrigido)...")
    print(f"Base Original: {DATA_DIR}")
    print(f"Base Recortada: {OUTPUT_DIR}")
    
    # 1. Encontrar todos os arquivos de imagem recursivamente
    img_files = []
    for root, dirs, files in os.walk(DATA_DIR):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_files.append(os.path.join(root, f))
                
    print(f"Total de imagens encontradas: {len(img_files)}")
    
    # Criar diretórios de classes no destino e na origem
    for class_name in class_names:
        os.makedirs(os.path.join(DATA_DIR, class_name), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, class_name), exist_ok=True)
        
    processed_count = 0
    
    for path in img_files:
        filename = os.path.basename(path)
        parent_dir = os.path.basename(os.path.dirname(path))
        
        # Determinar classe
        if parent_dir in class_names:
            class_name = parent_dir
        else:
            match_class = re.search(r'_a(\d{2})', filename)
            if match_class:
                class_code = match_class.group(1)
                class_name = code_to_class.get(class_code, 'Other')
            else:
                class_name = 'Other'
                
        # 2. Mover o arquivo original para a pasta de classe correta (se já não estiver lá)
        correct_parent_dir = os.path.join(DATA_DIR, class_name)
        new_original_path = os.path.join(correct_parent_dir, filename)
        
        if os.path.abspath(path) != os.path.abspath(new_original_path):
            shutil.move(path, new_original_path)
            current_path = new_original_path
        else:
            current_path = path
            
        # 3. Recortar a imagem e salvar no destino organizado
        filename_no_ext, _ = os.path.splitext(filename)
        dst_path = os.path.join(OUTPUT_DIR, class_name, f"{filename_no_ext}.png")
        
        # Ler imagem original
        img = cv2.imread(current_path)
        if img is None:
            continue
            
        # Extrair máscara corrigida
        mask = extract_object_mask(img)
        
        # Salvar como PNG transparente (RGBA)
        b, g, r = cv2.split(img)
        rgba = cv2.merge([b, g, r, mask])
        cv2.imwrite(dst_path, rgba)
        
        processed_count += 1
        if processed_count % 200 == 0:
            print(f"  -> {processed_count} imagens organizadas e recortadas...")
            
    print(f"\nFinalizado! Total de {processed_count} imagens organizadas e recortadas.")
    print(f"Imagens originais organizadas em: {DATA_DIR}")
    print(f"Imagens recortadas transparentes salvas em: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
