from flask import Flask, render_template, Response, jsonify, request
import cv2
import mediapipe as mp
import torch
import numpy as np
import os
import time
from datetime import datetime
from Modelo import ASLTranslator

app = Flask(__name__)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Variables globales
camera = None
texto_predicho = ""
ultima_prediccion = "nothing"
contador_estable = 0
ultima_prediccion_tiempo = time.time()
model = None

clases = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
          'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
          'space', 'del', 'nothing']

def cargar_modelo():
    """Carga el modelo entrenado."""
    try:
        model = ASLTranslator()
        model.load_state_dict(torch.load('mejor_modelo_fold_5.pth'))
        model.eval()
        print("Modelo cargado correctamente desde .pth")
        return model
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return None

def procesar_frame(frame, hands):
    """Procesa un frame para detectar manos y predecir la seña."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    hand_landmarks = None
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        
        mp_drawing.draw_landmarks(
            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2))
    
    return frame, hand_landmarks

def extraer_landmarks(landmarks):
    """Extrae los landmarks de la mano en formato adecuado para el modelo."""
    landmarks_list = []
    for lm in landmarks.landmark:
        landmarks_list.extend([lm.x, lm.y, lm.z])
    
    tensor = torch.tensor(landmarks_list, dtype=torch.float32).unsqueeze(0)
    return tensor

def predecir_seña(model, input_tensor):
    """Realiza una predicción con el modelo."""
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        predicted_idx = predicted.item()
        confidence = output[0][predicted_idx].item()
    
    return clases[predicted_idx], confidence

def guardar_predicciones(texto):
    """Guarda el texto traducido en un archivo."""
    if not os.path.exists('traducciones'):
        os.makedirs('traducciones')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"traduccion_{timestamp}.txt"
    filepath = os.path.join('traducciones', filename)
    with open(filepath, 'w') as f:
        f.write(texto)
    
    print(f"Texto guardado en {filepath}")
    return filepath

def generar_frames():
    """Generador que procesa los frames de la cámara para streaming."""
    global camera, model, texto_predicho, ultima_prediccion, contador_estable, ultima_prediccion_tiempo
    
    if camera is None:
        camera = cv2.VideoCapture(0)
    
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.7) as hands:
        
        while True:
            success, frame = camera.read()
            if not success:
                print("Error al leer de la cámara.")
                break
            
            frame = cv2.flip(frame, 1)
            
            frame, hand_landmarks = procesar_frame(frame, hands)
            
            if hand_landmarks and model is not None:
                input_tensor = extraer_landmarks(hand_landmarks)
                
                pred_class, confidence = predecir_seña(model, input_tensor)
                
                cv2.putText(frame, f"Predicción: {pred_class}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"Confianza: {confidence:.2f}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                tiempo_actual = time.time()
                
                if pred_class != "nothing":
                    if pred_class == ultima_prediccion:
                        contador_estable += 1
                        if contador_estable >= 15:
                            if tiempo_actual - ultima_prediccion_tiempo > 1:
                                if pred_class == "space":
                                    texto_predicho += " "
                                elif pred_class == "delete":
                                    texto_predicho = texto_predicho[:-1] if texto_predicho else ""
                                else:
                                    texto_predicho += pred_class
                                
                                contador_estable = 0
                                ultima_prediccion_tiempo = tiempo_actual
                    else:
                        ultima_prediccion = pred_class
                        contador_estable = 0
            else:
                ultima_prediccion = "nothing"
                contador_estable = 0
            
            cv2.putText(frame, "Texto:", (10, frame.shape[0] - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            max_chars_per_line = 40
            texto_wrapped = [texto_predicho[i:i+max_chars_per_line] 
                            for i in range(0, len(texto_predicho), max_chars_per_line)]
            
            for i, line in enumerate(texto_wrapped[-2:]):  # Mostrar solo las últimas 2 líneas
                cv2.putText(frame, line, (10, frame.shape[0] - 20 + i*30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """Ruta principal que muestra la interfaz web."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Ruta que proporciona el stream de video."""
    return Response(generar_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_text')
def get_text():
    """Ruta que devuelve el texto traducido actual."""
    global texto_predicho
    return jsonify({'texto': texto_predicho})

@app.route('/clear_text')
def clear_text():
    """Ruta para borrar el texto traducido."""
    global texto_predicho
    texto_predicho = ""
    return jsonify({'success': True})

@app.route('/save_text')
def save_text():
    """Ruta para guardar el texto traducido."""
    global texto_predicho
    if texto_predicho:
        filepath = guardar_predicciones(texto_predicho)
        return jsonify({'success': True, 'filepath': filepath})
    else:
        return jsonify({'success': False, 'error': 'No hay texto para guardar'})

@app.route('/shutdown', methods=['POST'])
def shutdown():
    """Cierra la cámara y libera recursos."""
    global camera
    if camera is not None:
        camera.release()
        camera = None
    return jsonify({'success': True})

if __name__ == '__main__':
    model = cargar_modelo()
    
    if model is None:
        print("Error: No se pudo cargar el modelo. Asegúrese de que exista el archivo mejor_modelo_asl.pth")
    else:
        print("Modelo cargado correctamente. Iniciando servidor web...")
        app.run(debug=True, host='0.0.0.0', port=5000)