import tensorflow as tf
import os
from data_processing import build_datasets
from model import build_model
import matplotlib.pyplot as plt
import pandas as pd


tf.config.optimizer.set_jit(True)

def main():
    print("A preparar o Pipeline de Dados")
    # Vai buscar os dados empacotados e prontos a entrar na rede
    train_ds, val_ds, num_ingredients, vocab = build_datasets()

    print(f"\nA construir o cerebro para {num_ingredients} ingredientes")
    model = build_model(num_ingredients)

    # Compilacao
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        
        # Como o modelo tem duas saídas (name='ingredientes' e name='peso'), 
        # temos de lhe dar duas formas de calcular o erro:
        loss={
            "ingredientes": "binary_crossentropy", # Puniçao pesada se falhar a probabilidade (0 a 1)
            "peso": "mae"                          # Mean Absolute Error: calcula o erro puro em gramas
        },
        
        # O erro das gramas pode ser 100 ou 200. O erro da probabilidade é tipo 0.5.
        # Se não multiplicarmos o peso por 0.01, o modelo só vai querer saber das gramas
        # e desiste de tentar adivinhar qual é o ingrediente
        loss_weights={
            "ingredientes": 1.0, 
            "peso": 0.01  
        },
        
        # O que queremos ver no ecrã para sabermos se está a correr bem:
        metrics={
            "ingredientes": [
                tf.keras.metrics.BinaryAccuracy(name="acc"),
                tf.keras.metrics.AUC(name="auc") # Excelente para ver se distingue bem o que esta/nao esta no prato
            ],
            "peso": "mae" # Mostra o erro médio em gramas na validação
        }
    )

    # Callbacks/ Checkpoints
    os.makedirs("checkpoints", exist_ok=True)
    callbacks = [
        # Guarda automaticamente a melhor versão
        tf.keras.callbacks.ModelCheckpoint(
            filepath="checkpoints/nutrifoods_best.keras",
            monitor="val_loss",      # Foca-se no erro da validação (dados que ele nunca viu)
            save_best_only=True,     # Só guarda se for melhor que a epoch anterior
            verbose=1
        ),
        # Fallback para evitar overfitting
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,              # Se passar 5 épocas sem melhorar, pára o treino.
            restore_best_weights=True,
            verbose=1
        ),
        # Cria logs
        tf.keras.callbacks.CSVLogger("logs/treino_log.csv", append=False)
    ]



    print("\nA iniciar o treino")
    try:
        # Treino
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=30,
            callbacks=callbacks
        )
        print("\nTreino finalizado com sucesso.")

    except KeyboardInterrupt:
        print("\nTreino interrompido pelo utilizador (Ctrl+C). A gerar logs parciais...")
        history = model.history # Recupera o que foi treinado até agora
    
    except Exception as e:
        print(f"\nOcorreu um erro inesperado: {e}")
        # Tenta recuperar o histórico se ele existir
        if hasattr(model, 'history'):
            history = model.history
        else:
            return # Sai se não houver mesmo nada para salvar

    # --- FASE DE LOGS (Executa mesmo se houver interrupção) ---
    try:
        print("A guardar métricas e gráficos...")
        os.makedirs("logs", exist_ok=True)
        
        # Guardar CSV
        hist_df = pd.DataFrame(history.history)
        hist_df.to_csv("logs/historico_final_detalhado.csv", index=False)

        # Criar gráficos
        plt.figure(figsize=(12, 5))
        
        # Gráfico 1: Loss
        plt.subplot(1, 2, 1)
        if 'loss' in history.history:
            plt.plot(history.history['loss'], label='Treino')
            plt.plot(history.history['val_loss'], label='Validação')
            plt.title('Evolução do Erro (Loss)')
            plt.legend()

        # Gráfico 2: Accuracy
        plt.subplot(1, 2, 2)
        if 'ingredientes_acc' in history.history:
            plt.plot(history.history['ingredientes_acc'], label='Treino')
            plt.plot(history.history['val_ingredientes_acc'], label='Validação')
            plt.title('Precisão dos Ingredientes')
            plt.legend()

        plt.tight_layout()
        plt.savefig("logs/grafico_treino.png")
        
        print("Logs e gráficos guardados na pasta 'logs/'.")
        
    except Exception as log_error:
        print(f"Erro ao gerar gráficos: {log_error}")

    print("\nProcesso concluído.")

if __name__ == "__main__":
    main()