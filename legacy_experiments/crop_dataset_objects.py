import os
import cv2
import numpy as np

DATA_DIR = r"c:\Users\Vinicius\Desktop\MestradoCodeAnti\Dataset_Wadaba"
OUTPUT_DIR = r"c:\Users\Vinicius\Desktop\MestradoCodeAnti\Dataset_Wadaba_Cropped"

def extract_object_mask(img):
    # Converter para escala de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # O fundo do WaDaBa é claro (geralmente > 220). Threshold binário invertido encontra o objeto.
    _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
    
    # Limpeza morfológica para remover ruído
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Encontrar o maior contorno (a garrafa/copo) para preencher buracos internos
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
        
    print(f"Iniciando o processamento do dataset para recorte de fundo...")
    print(f"Diretório Origem: {DATA_DIR}")
    print(f"Diretório Destino: {OUTPUT_DIR}")
    
    total_images_cropped = 0
    
    # Percorrer todas as pastas e arquivos do dataset original
    for root, dirs, files in os.walk(DATA_DIR):
        # Obter caminho relativo para recriar as mesmas subpastas no destino
        rel_path = os.path.relpath(root, DATA_DIR)
        target_folder = os.path.join(OUTPUT_DIR, rel_path)
        
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                if not os.path.exists(target_folder):
                    os.makedirs(target_folder)
                    
                src_path = os.path.join(root, file)
                
                # Definir nome de saída com extensão .png para suportar transparência (canal alpha)
                filename_no_ext, _ = os.path.splitext(file)
                dst_path = os.path.join(target_folder, f"{filename_no_ext}.png")
                
                # 1. Carregar imagem
                img = cv2.imread(src_path)
                if img is None:
                    continue
                    
                # 2. Extrair máscara de segmentação
                mask = extract_object_mask(img)
                
                # 3. Criar imagem de 4 canais (RGBA) com canal alpha transparente
                b, g, r = cv2.split(img)
                rgba = cv2.merge([b, g, r, mask])
                
                # 4. Salvar imagem recortada com transparência
                cv2.imwrite(dst_path, rgba)
                total_images_cropped += 1
                
                if total_images_cropped % 200 == 0:
                    print(f"  -> {total_images_cropped} imagens recortadas...")
                    
    print(f"\nConcluído! Total de {total_images_cropped} imagens processadas e salvas com fundo transparente em: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
