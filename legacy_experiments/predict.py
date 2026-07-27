import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Define as classes na mesma ordem do treinamento
# (Isso é muito importante: image_dataset_from_directory ordena as pastas alfabeticamente)
# No dataset temos: Other, PET, PE_HD, PP, PS
# Mas vamos deixar genérico (o train_models pode imprimir a ordem correta).
# Assumindo ordem alfabética que o Keras faz: ['Other', 'PET', 'PE_HD', 'PP', 'PS']
CLASS_NAMES = ['Other', 'PET', 'PE_HD', 'PP', 'PS']
IMG_SIZE = (224, 224)

def load_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(model_path, img_path):
    if not os.path.exists(model_path):
        print(f"Erro: Modelo não encontrado em {model_path}")
        return
        
    if not os.path.exists(img_path):
        print(f"Erro: Imagem não encontrada em {img_path}")
        return

    print(f"Carregando modelo {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    print(f"Carregando imagem {img_path}...")
    img_array = load_image(img_path)
    
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    confidence = predictions[0][predicted_class_index] * 100
    
    print("\n--- Resultado da Predição ---")
    print(f"Classe Prevista: {predicted_class_name}")
    print(f"Confiança: {confidence:.2f}%")
    print("-----------------------------\n")
    
    # Exibir probabilidade para cada classe
    print("Probabilidades por classe:")
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"{class_name}: {predictions[0][i]*100:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python predict.py <caminho_do_modelo.keras> <caminho_da_imagem.jpg>")
        print("Exemplo: python predict.py modelo_transfer.keras minha_imagem.jpg")
    else:
        modelo = sys.argv[1]
        imagem = sys.argv[2]
        predict_image(modelo, imagem)
