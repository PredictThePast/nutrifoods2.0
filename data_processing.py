import tensorflow as tf
import pandas as pd
import numpy as np
import os
import json
import csv
from sklearn.model_selection import train_test_split

# configs globais
IMG_SIZE = 300
BATCH_SIZE = 32
DATASET_DIR = "dataset/nutrition5k"
METADATA_DIR = os.path.join(DATASET_DIR, "metadata")
IMAGES_DIR = os.path.join(DATASET_DIR, "realsense_overhead")
def process_metadata():
    print("A analisar o CSV à prova de falhas (via módulo nativo csv)...")
    
    file_cafe1 = os.path.join(METADATA_DIR, "dish_metadata_cafe1.csv")
    file_cafe2 = os.path.join(METADATA_DIR, "dish_metadata_cafe2.csv")

    all_rows = []
    try:
        # le os dois ficheiros linha a linha, independentemente do seu tamanho
        for filepath in [file_cafe1, file_cafe2]:
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        all_rows.append(row)
            else:
                print(f"Aviso: Não encontrei o ficheiro {filepath}")
    except Exception as e:
        print(f"Erro ao ler os ficheiros CSV: {e}")
        return None, None, None, None, None

    image_paths = []
    ingredients_list = []
    weights_dict_list = []

    for row in all_rows:
        if not row: # salta linhas vazias
            continue
            
        dish_id = str(row[0]).strip()
        
        # Procurar a imagem
        img_path = os.path.join(IMAGES_DIR, dish_id, "rgb.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(IMAGES_DIR, dish_id, "rgb.png")
            if not os.path.exists(img_path):
                continue
            
        ing_names = []
        ing_weights = {}

        for i in range(6, len(row), 7):
            # se nao houver colunas suficientes para ler o nome e o peso, ou o ID for vazio, paramos
            if i + 2 >= len(row) or not row[i].strip():
                break
            
            nome_ingrediente = str(row[i+1]).strip()
            try:
                peso_ingrediente = float(row[i+2])
            except ValueError:
                peso_ingrediente = 0.0 # salvaguarda caso haja algum erro de formatação no texto
            
            ing_names.append(nome_ingrediente)
            ing_weights[nome_ingrediente] = peso_ingrediente
            
        image_paths.append(img_path)
        ingredients_list.append(ing_names)
        weights_dict_list.append(ing_weights)

    # ---------------------------------------------------------
    # CRIAR O VOCABULÁRIO E MATRIZES
    # ---------------------------------------------------------
    todas_as_tags = set(ing for sublist in ingredients_list for ing in sublist)
    vocabulario = sorted(list(todas_as_tags))
    num_ingredients = len(vocabulario)
    
    print(f"Encontrados {len(image_paths)} pratos válidos com {num_ingredients} ingredientes únicos.")

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/nutrition5k_vocab.json", "w") as f:
        json.dump(vocabulario, f, indent=2)

    multi_hot_labels = np.zeros((len(image_paths), num_ingredients), dtype=np.float32)
    multi_mass_labels = np.zeros((len(image_paths), num_ingredients), dtype=np.float32)

    for i, ing_names in enumerate(ingredients_list):
        for ing in ing_names:
            idx = vocabulario.index(ing)
            multi_hot_labels[i, idx] = 1.0
            multi_mass_labels[i, idx] = weights_dict_list[i][ing]

    return image_paths, multi_hot_labels, multi_mass_labels, num_ingredients, vocabulario

def load_and_preprocess_image(img_path, label_ing, label_weight):
    #redimensiona a imagem e coloca os rotulos
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # Preprocessamento específico da tua B0
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    
    # IMPORTANTE: Devolve um dicionário que corresponde aos nomes das camadas finais no model.py !!!!!!!!! 
    return img, {"ingredientes": label_ing, "peso": label_weight}


def augment(image, labels):
    # rodar imagens para simular varios angulos de uma camara de tlm
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image) 
    image = tf.image.random_brightness(image, 0.1)
    image = tf.image.random_contrast(image, 0.9, 1.1)
    
    # Roda a imagem aleatoriamente em 0, 90, 180 ou 270 graus
    random_rot = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k=random_rot)
    
    return image, labels


def build_datasets():
    # obter os dados processados do CSV
    paths, labels_ing, labels_weight, num_ingredients, vocab = process_metadata()
    
    # divisao de treino (80%) e validação (20%) usando o scikit-learn
    X_train, X_val, y_ing_train, y_ing_val, y_weight_train, y_weight_val = train_test_split(
        paths, labels_ing, labels_weight, test_size=0.2, random_state=42
    )
    
    print(f"Dados de Treino: {len(X_train)} | Dados de Validação: {len(X_val)}")

    # Criar as pipelines do TensorFlow
    AUTOTUNE = tf.data.AUTOTUNE
    
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_ing_train, y_weight_train))
    train_ds = train_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.map(augment, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE).prefetch(AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_ing_val, y_weight_val))
    val_ds = val_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    return train_ds, val_ds, num_ingredients, vocab


if __name__ == '__main__':
    try:
        print("A iniciar o pipeline de processamento de dados...")
        train_ds, val_ds, num_ingredients, vocab = build_datasets()
        print("\n O pipeline de dados foi criado com sucesso.")
    except Exception as e:
        print(f"\n Falha ao construir os datasets.")
        print(f"Detalhes do erro: {e}")