import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow as tf
from tensorflow.keras import layers, models

import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(num_ingredients, img_size=300): # Atualizado para 300 para mais detalhe
    print(f"A construir modelo multi-saída para {num_ingredients} ingredientes")

    # Backbone: EfficientNetB0
    # include_top=False para nao usarmos a cabeça original de 1000 classes da Google
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights='imagenet'
    )

    # ATIVAMOS O FINE-TUNING: Desbloqueamos o modelo base para a máxima potência.
    # Isto permite que a rede ajuste o que já sabe da Google para o teu caso específico de comida.
    base_model.trainable = True 

    # Camada de ligacao
    # Transforma o mapa de características 2D num vetor 1D que o cérebro entende
    x = layers.GlobalAveragePooling2D()(base_model.output)
    
    # Uma camada intermedia para ajudar a processar a informação antes da divisão. 512 neuronios
    x = layers.Dense(512, activation='relu')(x)
    
    # Evita que o modelo decore os dados (overfitting). desliga 30% dos neuronios aleatoriamente
    x = layers.Dropout(0.3)(x) 

    # ---------------------------------------------------------
    # CABEÇA 1: Classificação de Ingredientes (Probabilidades)
    # ---------------------------------------------------------
    # Usamos sigmoid porque um prato pode ter VÁRIOS ingredientes ao mesmo tempo
    out_ingredientes = layers.Dense(
        num_ingredients, 
        activation='sigmoid', # sigmoid garante que o numero que o modelo devolve esta dentro de 0 e 1.
                              # Ao contrario de por exemplo, o softmax que so pode ter uma classe a 99%, 
                              # o sigmoid permite varias. Ideal para classificar varios ingredientes (Multi-label)
        name='ingredientes' # <--- Este nome tem de bater com o train.py e data_processing.py
    )(x)

    # ---------------------------------------------------------
    # CABEÇA 2: Regressão de Pesos Individuais (Gramas)
    # ---------------------------------------------------------
    # Usamos relu porque o peso nunca pode ser negativo (mínimo 0g)
    out_peso = layers.Dense(
        num_ingredients, 
        activation='relu', # Rectified Linear Unit <=> f(x) = max(0, x); ou seja, devolve sempre o valor maior. 
                           # Se o valor for negativo devolve 0. Se o valor for positivo, devolve o valor.
        name='peso' # <--- Este nome tem de bater com o train.py e data_processing.py
    )(x)

    # UNIR TUDO. declarar o model completo
    model = models.Model(
        inputs=base_model.input, 
        outputs=[out_ingredientes, out_peso]
    )

    print("Arquitetura do modelo finalizada com sucesso")
    return model


# Pequeno teste ao correr o ficheiro sozinho. se importar, ignora esta parte. o teste so funciona no terminal
if __name__ == "__main__":
    m = build_model(100) # Simula 100 ingredientes
    m.summary() # Mostra a estrutura da rede no terminal

    # guarda a estrutura da rede num .txt
    with open('arquitetura_modelo.txt', 'w') as f:
        m.summary(print_fn=lambda x: f.write(x + '\n'))
    
    print("Tabela guardada no ficheiro 'arquitetura_modelo.txt'")