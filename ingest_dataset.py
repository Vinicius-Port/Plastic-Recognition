import os
import re
import shutil
import hashlib
import argparse
from PIL import Image
from collections import defaultdict

# Mapeamento de Classes para Códigos de Arquivo do WaDaBa
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

VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

def validate_image(file_path):
    """
    Verifica se a imagem é válida, não corrompida e tem tamanho > 0 bytes.
    Retorna True/False e o objeto PIL.Image ou None.
    """
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return False, "Arquivo vazio ou não encontrado"
    
    try:
        with Image.open(file_path) as img:
            img.verify()  # Verifica integridade do arquivo
        
        # Abre novamente para manipulação (verify invalida a instância)
        with Image.open(file_path) as img:
            img_rgb = img.convert('RGB')
            return True, img_rgb
    except Exception as e:
        return False, str(e)

def generate_wadaba_filename(object_id, class_name, file_path):
    """
    Gera o nome de arquivo padronizado no formato WaDaBa: <ID_OBJETO>_a<CODIGO_CLASSE>_<HASH>.<EXT>
    Exemplo: 0105_a01b01c2d0e0f1g0h3.jpg
    """
    clean_obj_id = str(object_id).zfill(4)
    clean_class = class_name.upper().replace("-", "_")
    code = CLASS_TO_CODE.get(clean_class, '07')
    
    with open(file_path, 'rb') as f:
        file_hash = hashlib.md5(f.read()).hexdigest()[:8]
        
    _, ext = os.path.splitext(file_path)
    if not ext or ext.lower() not in VALID_EXTENSIONS:
        ext = '.jpg'
        
    new_filename = f"{clean_obj_id}_a{code}b01c2d0e0f1_{file_hash}{ext.lower()}"
    return new_filename

def ingest_single_image(input_file, target_dataset_dir, object_id, class_name, move=False):
    """
    Ingere e padroniza uma única imagem no dataset.
    """
    is_valid, result = validate_image(input_file)
    if not is_valid:
        print(f"[ERRO] Imagem inválida '{input_file}': {result}")
        return False
    
    clean_class_dir = CODE_TO_CLASS.get(CLASS_TO_CODE.get(class_name.upper().replace("-", "_"), '07'), 'Other')
    dest_dir = os.path.join(target_dataset_dir, clean_class_dir)
    os.makedirs(dest_dir, exist_ok=True)
    
    new_filename = generate_wadaba_filename(object_id, clean_class_dir, input_file)
    dest_path = os.path.join(dest_dir, new_filename)
    
    result.save(dest_path, format="JPEG", quality=95)
    
    if move:
        os.remove(input_file)
        print(f"[INGESTÃO] Imagem movida e padronizada -> {dest_path}")
    else:
        print(f"[INGESTÃO] Imagem copiada e padronizada -> {dest_path}")
        
    return dest_path

def ingest_batch_directory(incoming_dir, target_dataset_dir, default_object_id=None, default_class=None):
    """
    Ingere em lote todas as imagens de um diretório de entrada.
    """
    if not os.path.exists(incoming_dir):
        print(f"[ERRO] Pasta de entrada '{incoming_dir}' não existe.")
        return 0
    
    count = 0
    obj_counter = int(default_object_id) if default_object_id else 9000
    
    for root, _, files in os.walk(incoming_dir):
        for f in files:
            if f.lower().endswith(VALID_EXTENSIONS):
                file_path = os.path.join(root, f)
                parent_name = os.path.basename(root)
                
                detected_class = default_class
                if not detected_class:
                    if parent_name.upper().replace("-", "_") in CLASS_TO_CODE:
                        detected_class = parent_name
                    else:
                        detected_class = 'Other'
                        
                match_obj = re.match(r'^(\d+)', f)
                if match_obj:
                    obj_id = match_obj.group(1)
                else:
                    obj_id = obj_counter
                    obj_counter += 1
                    
                ingest_single_image(file_path, target_dataset_dir, obj_id, detected_class, move=False)
                count += 1
                
    print(f"\n[SUCESSO] Ingestão em lote concluída: {count} imagens processadas em '{target_dataset_dir}'.")
    return count

POSSIBLE_DATA_DIRS = [
    "./Dataset_Wadaba",
    "../Datasets/Dataset_Wadaba",
    "../Dataset_Wadaba",
    r"C:\Users\Vinicius\Desktop\MestradoCodeAnti\ProjetoPlasticRecognition\Datasets\Dataset_Wadaba",
    "/kaggle/input/datasets/vinicius1portugal/dataset-wadaba/Dataset_modificado",
    "/kaggle/input/datasets/vinicius1portugal/dataset-wadaba",
    "/kaggle/input/dataset-wadaba/Dataset_Wadaba",
    "/kaggle/input/dataset-wadaba"
]

def resolve_dataset_dir(requested_dir=None):
    if requested_dir and os.path.exists(requested_dir):
        return requested_dir
    for cand in POSSIBLE_DATA_DIRS:
        if cand and os.path.exists(cand):
            return cand
    return requested_dir if requested_dir else "./Dataset_Wadaba"

def run_quality_gate(dataset_dir):
    dataset_dir = resolve_dataset_dir(dataset_dir)
    """
    Verifica a qualidade e o estado do dataset.
    """
    print(f"\n=======================================================")
    print(f"DATA QUALITY GATE: Analisando '{dataset_dir}'")
    print(f"=======================================================")
    
    if not os.path.exists(dataset_dir):
        print(f"[ERRO CRÍTICO] Diretório do dataset não encontrado: {dataset_dir}")
        return False
        
    class_counts = defaultdict(int)
    object_counts = defaultdict(set)
    corrupted_count = 0
    total_files = 0
    
    for root, _, files in os.walk(dataset_dir):
        for f in files:
            if f.lower().endswith(VALID_EXTENSIONS):
                total_files += 1
                file_path = os.path.join(root, f)
                
                is_valid, _ = validate_image(file_path)
                if not is_valid:
                    print(f"[AVISO] Arquivo corrompido encontrado: {file_path}")
                    corrupted_count += 1
                    continue
                
                parent_dir = os.path.basename(os.path.dirname(file_path))
                class_counts[parent_dir] += 1
                
                match_obj = re.match(r'^(\d+)', f)
                if match_obj:
                    object_counts[parent_dir].add(match_obj.group(1))
                    
    print(f"Total de Imagens Válidas: {total_files - corrupted_count} / {total_files}")
    print(f"Imagens Corrompidas: {corrupted_count}")
    print("\nDistribuição por Classe e Objetos Únicos (LOOO):")
    for cls_name, cnt in class_counts.items():
        objs = len(object_counts[cls_name])
        print(f"  - {cls_name:<10}: {cnt:>5} imagens | {objs:>3} objetos físicos distintos")
        
    print("=======================================================\n")
    return corrupted_count == 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Ingestão e Validação do Dataset WaDaBa")
    parser.add_argument("--incoming", type=str, default=None, help="Diretório com novas imagens para ingerir")
    parser.add_argument("--dataset_dir", type=str, default="./Dataset_Wadaba", help="Diretório alvo do dataset")
    parser.add_argument("--obj_id", type=str, default=None, help="ID do objeto (ex: 0105)")
    parser.add_argument("--class_name", type=str, default=None, help="Classe do plástico (PET, PE_HD, PP, PS, Other)")
    parser.add_argument("--check_quality", action="store_true", help="Executar apenas o Data Quality Gate")
    
    args = parser.parse_args()
    
    if args.check_quality or not args.incoming:
        run_quality_gate(args.dataset_dir)
    elif args.incoming:
        ingest_batch_directory(args.incoming, args.dataset_dir, args.obj_id, args.class_name)
        run_quality_gate(args.dataset_dir)
