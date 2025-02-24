import cv2
import joblib
import numpy as np

# Muat model SVM serta scaler
svm_model = joblib.load('data_svm.pkl')
scaler = joblib.load('scaler.pkl')

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    height, width, _ = frame.shape
    
    # Tentukan beberapa titik sampel dalam frame
    sample_points = [
        (height // 2, width // 2),  # Tengah
        (height // 4, width // 4),  # Kiri atas
        (height // 4, 3 * width // 4),  # Kanan atas
        (3 * height // 4, width // 4),  # Kiri bawah
        (3 * height // 4, 3 * width // 4)  # Kanan bawah
    ]
    
    colors_detected = []
    for (y, x) in sample_points:
        pixel = frame[y, x]
        pixel_scaled = scaler.transform([pixel])
        color_pred = svm_model.predict(pixel_scaled)[0]
        colors_detected.append((x, y, color_pred))
    
    # Tampilkan warna yang terdeteksi di layar
    for (x, y, color) in colors_detected:
        cv2.putText(frame, f'{color}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.circle(frame, (x, y), 5, (0, 0, 0), -1)  # Tandai titik dengan lingkaran
    
    cv2.imshow('Frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()