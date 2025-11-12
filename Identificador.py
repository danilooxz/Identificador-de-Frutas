from tensorflow.keras.models import load_model
import numpy as np
import cv2

# Carrega o modelo
model = load_model("keras_model.h5", compile=False)

# Carrega os labels
labels = [line.strip() for line in open("labels.txt", "r", encoding="utf-8")]

# Inicia a webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (224, 224))
    img = np.expand_dims(img / 255.0, axis=0)

    prediction = model.predict(img)
    index = np.argmax(prediction)
    label = labels[index]
    confidence = prediction[0][index]

    cv2.putText(frame, f"{label} ({confidence:.2f})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Reconhecimento de Frutas", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()