import os
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
from PIL import Image
import rembg

def process_single_image(args_tuple):
    src_path, dst_path, session = args_tuple
    if os.path.exists(dst_path):
        return True # Já processada

    try:
        orig_img = Image.open(src_path).convert("RGB")
        
        # Remove o fundo da esteira e obtém máscara RGBA via U^2-Net
        rgba_img = rembg.remove(orig_img, session=session)
        rgba_np = np.array(rgba_img)

        # Separa canais RGB e canal Alpha (0 a 255)
        rgb = rgba_np[:, :, :3]
        alpha = rgba_np[:, :, 3] / 255.0

        # Cria fundo 100% preto neutro
        black_bg = np.zeros_like(rgb, dtype=np.float32)

        # Fusão Alpha: Objeto * Alpha + FundoPreto * (1 - Alpha)
        masked_obj = (rgb.astype(np.float32) * alpha[:, :, np.newaxis] + black_bg * (1.0 - alpha[:, :, np.newaxis])).astype(np.uint8)

        # Salva imagem resultante
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        cv2.imwrite(dst_path, cv2.cvtColor(masked_obj, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        return True
    except Exception as e:
        print(f"[ERRO ao processar {os.path.basename(src_path)}]: {e}")
        return False

def generate_masked_dataset(input_dir, output_dir, max_workers=4):
    print("==========================================================================")
    print("GERADOR DE DATASET COM FUNDO NEUTRO MASCARADO (U^2-NET)")
    print(f"Entrada: {input_dir}")
    print(f"Saída:   {output_dir}")
    print("==========================================================================")

    session = rembg.new_session("u2net")
    classes = ['PET', 'PE_HD', 'PP', 'PS', 'Other']
    
    tasks = []
    total_imgs = 0

    for cls in classes:
        cls_dir = os.path.join(input_dir, cls)
        if not os.path.exists(cls_dir):
            continue

        out_cls_dir = os.path.join(output_dir, cls)
        os.makedirs(out_cls_dir, exist_ok=True)

        img_files = [f for f in sorted(os.listdir(cls_dir)) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        total_imgs += len(img_files)

        for f in img_files:
            src = os.path.join(cls_dir, f)
            dst = os.path.join(out_cls_dir, f)
            tasks.append((src, dst, session))

    print(f"\n[INFO] Total de {len(tasks)} imagens encontradas para mascarar o fundo.")

    success_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, res in enumerate(executor.map(process_single_image, tasks)):
            if res:
                success_count += 1
            if (idx + 1) % 100 == 0 or (idx + 1) == len(tasks):
                print(f"  - Progresso: [{idx + 1:04d}/{len(tasks):04d}] imagens mascaradas com sucesso...")

    print(f"\n==========================================================================")
    print(f"🎉 Concluído! {success_count} imagens mascaradas salvas em:")
    print(f"📂 {output_dir}")
    print("==========================================================================\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Gera versão mascarada com fundo preto neutro para eliminar viés de esteira.")
    default_in = r"C:\Users\Vinicius\Desktop\MestradoCodeAnti\Datasets\Dataset_Wadaba_PlusOther"
    default_out = r"C:\Users\Vinicius\Desktop\MestradoCodeAnti\Datasets\Dataset_Wadaba_MaskedBlack"
    parser.add_argument("--input_dir", type=str, default=default_in, help="Caminho do dataset original com 5 classes")
    parser.add_argument("--output_dir", type=str, default=default_out, help="Caminho onde salvar o dataset com fundo preto")
    parser.add_argument("--workers", type=int, default=4, help="Número de threads simultâneas")

    args = parser.parse_args()
    generate_masked_dataset(args.input_dir, args.output_dir, args.workers)
