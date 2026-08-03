import os
import cv2
import re
import shutil
import hashlib
import argparse
import numpy as np
from PIL import Image

CLASS_TO_CODE = {
    'PET': '01',
    'PE_HD': '02',
    'PP': '05',
    'PS': '06',
    'OTHER': '07',
    'OUTROS': '07'
}

CODE_TO_CLASS = {
    '01': 'PET',
    '02': 'PE_HD',
    '05': 'PP',
    '06': 'PS',
    '07': 'Other'
}

def remove_and_replace_background(image_path, texture_path, output_path):
    """
    Remove o fundo branco/claro de uma imagem de objeto e substitui
    pela textura real da esteira rolante (esteira_textura.jpg).
    """
    img = cv2.imread(image_path)
    bg_texture = cv2.imread(texture_path)

    if img is None:
        print(f"[ERRO] Não foi possível ler a imagem do objeto: {image_path}")
        return False

    if bg_texture is None:
        print(f"[ERRO] Não foi possível ler a imagem da textura: {texture_path}")
        return False

    h, w = img.shape[:2]

    if bg_texture.shape[0] < h or bg_texture.shape[1] < w:
        bg_texture = cv2.resize(bg_texture, (w, h), interpolation=cv2.INTER_CUBIC)
    else:
        bg_texture = bg_texture[:h, :w]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    white_mask = (gray > 225) & (hsv[:, :, 1] < 45)

    object_mask = np.where(white_mask, 0, 255).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    alpha_mask = cv2.GaussianBlur(object_mask, (7, 7), 0).astype(float) / 255.0
    alpha_3d = cv2.merge([alpha_mask, alpha_mask, alpha_mask])

    foreground = img.astype(float) * alpha_3d
    background = bg_texture.astype(float) * (1.0 - alpha_3d)
    combined = cv2.add(foreground, background).clip(0, 255).astype(np.uint8)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, combined)
    return True

def generate_wadaba_filename(object_id, class_name, input_path):
    """
    Gera o nome no padrão WaDaBa oficial para qualquer classe.
    Exemplo para PET (a01): 5001_a01b01c2d0e0f1_a1b2c3d4.jpg
    Exemplo para Other (a07): 5001_a07b01c2d0e0f1_a1b2c3d4.jpg
    """
    clean_obj_id = str(object_id).zfill(4)
    clean_class = class_name.upper().replace("-", "_")
    code = CLASS_TO_CODE.get(clean_class, '07')

    with open(input_path, 'rb') as f:
        file_hash = hashlib.md5(f.read()).hexdigest()[:8]
    _, ext = os.path.splitext(input_path)
    if not ext or ext.lower() not in ('.jpg', '.jpeg', '.png'):
        ext = '.jpg'
    return f"{clean_obj_id}_a{code}b01c2d0e0f1_{file_hash}{ext.lower()}"

def process_white_bg_directory(incoming_dir, target_dataset_dir, texture_path, class_name="Other", start_obj_id=5000):
    """
    Processa em lote uma pasta de imagens de fundo branco destinadas a uma classe específica,
    substituindo o fundo e registrando no dataset oficial WaDaBa.
    """
    if not os.path.exists(incoming_dir):
        print(f"[ERRO] Diretório de imagens de entrada não existe: {incoming_dir}")
        return 0

    if not os.path.exists(texture_path):
        print(f"[ERRO] Textura da esteira não encontrada: {texture_path}")
        return 0

    clean_class_dir = CODE_TO_CLASS.get(CLASS_TO_CODE.get(class_name.upper().replace("-", "_"), '07'), 'Other')
    dest_class_dir = os.path.join(target_dataset_dir, clean_class_dir)
    os.makedirs(dest_class_dir, exist_ok=True)

    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    count = 0
    obj_counter = start_obj_id

    for root, _, files in os.walk(incoming_dir):
        for f in files:
            if f.lower().endswith(valid_exts):
                src_path = os.path.join(root, f)

                match_obj = re.match(r'^(\d+)', f)
                if match_obj:
                    obj_id = match_obj.group(1)
                else:
                    obj_id = obj_counter
                    obj_counter += 1

                new_filename = generate_wadaba_filename(obj_id, clean_class_dir, src_path)
                dest_path = os.path.join(dest_class_dir, new_filename)

                success = remove_and_replace_background(src_path, texture_path, dest_path)
                if success:
                    count += 1
                    print(f"[INGESTÃO {clean_class_dir.upper()}] Imagem processada com fundo de esteira -> {dest_path}")

    print(f"\n[SUCESSO] Processamento de '{clean_class_dir}' concluído: {count} imagens ingeridas em '{dest_class_dir}'.")
    return count

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Ferramenta de Substituição de Fundo para o Dataset")
    parser.add_argument("--incoming", type=str, required=True, help="Diretório com novas imagens de fundo branco")
    parser.add_argument("--dataset_dir", type=str, default="../Datasets/Dataset_Wadaba", help="Diretório do dataset WaDaBa")
    parser.add_argument("--class_name", type=str, default="Other", help="Classe de destino (PET, PE_HD, PP, PS, Other)")
    parser.add_argument("--texture", type=str, default="./data/esteira_textura.jpg", help="Caminho para esteira_textura.jpg")
    parser.add_argument("--start_id", type=int, default=5000, help="ID inicial do objeto para LOOO")

    args = parser.parse_args()
    process_white_bg_directory(args.incoming, args.dataset_dir, args.texture, args.class_name, args.start_id)
