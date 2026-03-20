import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(num_ingredients, img_size=300): 
    print(f"A construir modelo multi-saída para {num_ingredients} ingredientes")

    # Backbone: EfficientNetB0
    # include_top=False para nao usarmos a cabeça original de 1000 classes da Google
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights='imagenet'
    )

    # Bloqueamos o modelo base no início (Fine-Tuning Seguro).
    base_model.trainable = False 

    # Camada de ligação (Pooling comum)
    x = layers.GlobalAveragePooling2D()(base_model.output)

    # =========================================================
    # RAMO 1: Classificação de Ingredientes (Nomes)
    # =========================================================
    # Uma parte do "cérebro" dedicada a cores e texturas
    x_class = layers.Dense(512, activation='relu')(x)
    x_class = layers.Dropout(0.3)(x_class)
    
    out_ingredientes = layers.Dense(
        num_ingredients, 
        activation='sigmoid', # sigmoid garante que o numero que o modelo devolve esta dentro de 0 e 1.
                              # Ideal para classificar varios ingredientes (Multi-label)
        name='ingredientes'   # <--- Este nome tem de bater com o train.py e data_processing.py
    )(x_class)

    # =========================================================
    # RAMO 2: Regressão de Pesos Individuais (Gramas)
    # =========================================================
    # Outra parte do "cérebro" dedicada a área e volume
    x_weight = layers.Dense(512, activation='relu')(x)
    x_weight = layers.Dropout(0.3)(x_weight)
    
    out_peso = layers.Dense(
        num_ingredients, 
        activation='softplus', # softplus evita que os neuronios morram (como na relu) com muitos pesos a 0g
        name='peso'            # <--- Este nome tem de bater com o train.py e data_processing.py
    )(x_weight)

    # UNIR TUDO. Declarar o model completo
    model = models.Model(
        inputs=base_model.input, 
        outputs=[out_ingredientes, out_peso]
    )

    print("Arquitetura do modelo finalizada com sucesso")
    return model, base_model

# Pequeno teste ao correr o ficheiro sozinho.
if __name__ == "__main__":
    m = build_model(241) 
    m.summary() # Mostra a estrutura da rede no terminal

    # guarda a estrutura da rede num .txt
    with open('arquitetura_modelo.txt', 'w') as f:
        m.summary(print_fn=lambda x: f.write(x + '\n'))
    
    print("Tabela guardada no ficheiro 'arquitetura_modelo.txt'")