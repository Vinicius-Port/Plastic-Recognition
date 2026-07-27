import os
import cv2
import shutil
import numpy as np

DATA_DIR = r"C:\Users\Vinicius\Desktop\MestradoCodeAnti\ProjetoPlasticRecognition\Datasets\Dataset_Wadaba"
OUTPUT_DIR = r"C:\Users\Vinicius\Desktop\MestradoCodeAnti\ProjetoPlasticRecognition\Datasets\Dataset_Wadaba_Cropped_Convex"

def extract_object_mask_convex(img):
    # Converter para escala de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Otsu's thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Limpeza morfológica
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mask = np.zeros_like(thresh)
    if contours:
        c = max(contours, key=cv2.contourArea)
        # APLICAÇÃO DO CONVEX HULL: Cria uma envoltória elástica que fecha todos os "buracos" de transparência
        hull = cv2.convexHull(c)
        cv2.drawContours(mask, [hull], -1, 255, -1)
    else:
        mask = thresh
        
    return mask

def main():
    if not os.path.exists(DATA_DIR):
        print(f"ERRO: Diretório de dados {DATA_DIR} não encontrado.")
        return
        
    class_names = ['Other', 'PET', 'PE_HD', 'PP', 'PS']
    
    print("Iniciando recorte de dataset usando CONVEX HULL...")
    print(f"Base Original: {DATA_DIR}")
    print(f"Base Recortada (Convex Hull): {OUTPUT_DIR}")
    
    # Encontrar todos os arquivos de imagem recursivamente
    img_files = []
    for root, dirs, files in os.walk(DATA_DIR):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_files.append(os.path.join(root, f))
                
    print(f"Total de imagens encontradas: {len(img_files)}")
    
    # Criar diretórios de classes no destino
    for class_name in class_names:
        os.makedirs(os.path.join(OUTPUT_DIR, class_name), exist_ok=True)
        
    processed_count = 0
    
    for path in img_files:
        filename = os.path.basename(path)
        parent_dir = os.path.basename(os.path.dirname(path))
        
        # Ignorar se o parent_dir não for uma classe (segurança extra se houver arquivos soltos)
        if parent_dir not in class_names:
            continue
            
        filename_no_ext, _ = os.path.splitext(filename)
        dst_path = os.path.join(OUTPUT_DIR, parent_dir, f"{filename_no_ext}.png")
        
        # Pular se a imagem já existir para evitar reprocessamento
        if os.path.exists(dst_path):
            processed_count += 1
            continue
            
        img = cv2.imread(path)
        if img is None:
            continue
            
        # Extrair máscara convexa
        mask = extract_object_mask_convex(img)
        
        # Salvar como PNG com canal alpha de transparência
        b, g, r = cv2.split(img)
        rgba = cv2.merge([b, g, r, mask])
        cv2.imwrite(dst_path, rgba)
        
        processed_count += 1
        if processed_count % 200 == 0:
            print(f"  -> {processed_count} imagens recortadas com Convex Hull...")
            
    print(f"\nFinalizado! Total de {processed_count} imagens recortadas e salvas em: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
