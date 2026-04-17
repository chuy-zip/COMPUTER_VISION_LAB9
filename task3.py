# task 3
# integrantes

# - Sergio Orellana 221122
# - Rodrigo Mansilla 22611
# - Ricardo Chuy 221007

import cv2
import time
from ultralytics import YOLO

# Modelo: YOLOv8n (nano) pre-entrenado para mientras solo de prueba
MODEL_PATH = "yolov8n.pt"

# Hiperparámetros de inferencia se elige una confianza minim que sirve para aceptar una deteccion
# 
CONF_THRESHOLD = 0.5   # Confianza mínima para aceptar una detección
IOU_THRESHOLD  = 0.45  # Umbral NMS: suprime cajas con IoU >= este valor

# la fuente de vide si es 0 = webcam principa, para usar MP4 cuando esté disponible: cambiar a "ruta/al/video.mp4"
VIDEO_SOURCE = 0

# Colores BGR para las cajas y texto
COLOR_BOX  = (0, 255, 0)    
COLOR_FPS  = (0, 0, 255)    
COLOR_TEXT = (255, 255, 255) 

model = YOLO(MODEL_PATH)
cap   = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    raise RuntimeError(f"No se pudo abrir la fuente de video: {VIDEO_SOURCE}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # inferencia y medicion fps
    t_start = time.time()
    results  = model(frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
    t_end   = time.time()

    fps = 1.0 / (t_end - t_start)

    # Extracción de tensores y dibujo manual
    # results[0].boxes.xyxy → tensor shape [N, 4] con cord absolutas (x1,y1,x2,y2)
    # results[0].boxes.conf → tensor shape [N]    con score de confianza
    # results[0].boxes.cls  → tensor shape [N]    con id de clase
    boxes = results[0].boxes

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
        conf_score      = float(boxes.conf[i].item())
        cls_id          = int(boxes.cls[i].item())
        label           = model.names[cls_id]

        # bouding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BOX, 2)

        # etiqueta con fondo negro
        text     = f"{label} {conf_score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), (0, 0, 0), -1)
        cv2.putText(frame, text, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 1, cv2.LINE_AA)

    # lo fps
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_FPS, 2, cv2.LINE_AA)

    cv2.imshow("Pokedex CV - COCO | presiona Q para salir", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
