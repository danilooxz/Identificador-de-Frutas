# Importa as bibliotecas necessárias
from tensorflow.keras.models import load_model  # Para carregar o modelo de rede neural previamente treinado
import numpy as np                              # Para operações matemáticas e manipulação de arrays
import cv2                                      # OpenCV: usada para capturar vídeo da webcam e exibir imagens

# Carrega o modelo treinado salvo no arquivo 'keras_model.h5'
# O parâmetro compile=False indica que o modelo não será recompilado.
model = load_model("keras_model.h5", compile=False)

# Abre o arquivo 'labels.txt', que contém os nomes das classes (ex: "Maçã", "Banana", etc.)
# O strip() remove espaços e quebras de linha no início e fim de cada linha.
labels = [line.strip() for line in open("labels.txt", "r", encoding="utf-8")]

# Cria um objeto que captura o vídeo da webcam padrão
cap = cv2.VideoCapture(0)

while True:
    # Captura um frame (imagem) da webcam
    # ret indica se a captura foi bem-sucedida (True/False)
    # frame contém a imagem capturada
    ret, frame = cap.read()
    
    # Se a captura falhar (ret=False), sai do loop
    if not ret:
        break
    
    # Redimensiona a imagem para o tamanho esperado pelo modelo (224x224 pixels)
    img = cv2.resize(frame, (224, 224))
    
    # O modelo espera uma entrada no formato (1, 224, 224, 3)
    img = np.expand_dims(img / 255.0, axis=0)       

    # Faz a predição usando o modelo carregado
    # O resultado é um array com as probabilidades de cada classe
    prediction = model.predict(img)
    
    # np.argmax() retorna o índice da classe com maior probabilidade
    index = np.argmax(prediction)
    
    # Obtém o nome da classe correspondente ao índice
    label = labels[index]
    
    # Obtém o valor da confiança (probabilidade) da classe predita
    confidence = prediction[0][index]

    # Escreve o nome da classe e a confiança na imagem original capturada da webcam
    # Posição do texto: (10, 30)
    # Fonte: FONT_HERSHEY_SIMPLEX
    # Cor: verde (0, 255, 0)
    # Espessura: 2
    cv2.putText(frame, f"{label} ({confidence:.2f})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Mostra a imagem com o rótulo em uma janela chamada "Reconhecimento de Frutas"
    cv2.imshow("Reconhecimento de Frutas", frame)

    # Se o usuário pressionar a tecla 'q', o loop é encerrado
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera o acesso à webcam
cap.release()

# Fecha todas as janelas abertas pelo OpenCV
cv2.destroyAllWindows()
