# scripts/detect_realtime.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('../model/quality_model.h5')
classes = ['good', 'defective']

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    img = cv2.resize(frame, (128, 128)) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    label = classes[np.argmax(prediction)]

    color = (0, 255, 0) if label == 'good' else (0, 0, 255)
    cv2.putText(frame, f"Result: {label}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Visual Quality Check", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
