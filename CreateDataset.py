import numpy as np
import os
import cv2
import mediapipe as mp
import torch
from torch.utils.data import Dataset
import pickle
import time

clases = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
          'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
          'space', 'del']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class ASLDataset(Dataset):
    def __init__(self, data_dir=None, processed_data_path=None, transform=None,file = "asl_dataset_processed_2.pkl"):
        """
        Inicializa el dataset de ASL.
        
        Args:
            data_dir (str, optional): Directorio con las imágenes organizadas por clases.
            processed_data_path (str, optional): Ruta a un archivo de datos preprocesados.
            transform (callable, optional): Transformación opcional a aplicar a las muestras.
        """
        self.transform = transform
        self.samples = []
        self.labels = []

        
        if processed_data_path and os.path.exists(processed_data_path):
            self._load_processed_data(os.path.join(processed_data_path, file))
        elif data_dir:
            self._process_images(data_dir)
        else:
            raise ValueError("Debe proporcionar data_dir o processed_data_path")
    
    def _process_images(self, data_dir):
        """Procesa las imágenes desde el directorio y extrae los landmarks."""
        self.hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
        
        total_files = sum([len(files) for _, _, files in os.walk(data_dir) if len(files) > 0])
        processed_files = 0
        start_time = time.time()
        
        for class_idx, class_name in enumerate(clases):
            class_path = os.path.join(data_dir, class_name)
            if os.path.isdir(class_path):
                print(f"Procesando clase: {class_name}")
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    try:
                        img = cv2.imread(img_path)
                        if img is None:
                            continue
                        
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        
                        results = self.hands.process(img_rgb)
                        
                        if results.multi_hand_landmarks:
                            landmarks = []
                            for hand_landmarks in results.multi_hand_landmarks:
                                for lm in hand_landmarks.landmark:
                                    landmarks.extend([lm.x, lm.y, lm.z])
                            
                            self.samples.append(np.array(landmarks, dtype=np.float32))
                            self.labels.append(class_idx)
                        
                        processed_files += 1
                        if processed_files % 100 == 0:
                            elapsed_time = time.time() - start_time
                            estimated_total = elapsed_time * (total_files / processed_files)
                            remaining_time = estimated_total - elapsed_time
                            print(f"Progreso: {processed_files}/{total_files} ({processed_files/total_files*100:.1f}%) - "
                                  f"Tiempo restante estimado: {remaining_time/60:.1f} minutos")
                    
                    except Exception as e:
                        print(f"Error procesando {img_path}: {e}")
        
        print(f"Dataset procesado: {len(self.samples)} muestras de {processed_files} imágenes")
        print(f"Tiempo total de procesamiento: {(time.time() - start_time)/60:.2f} minutos")
    
    def _load_processed_data(self, file_path):
        """Carga los datos procesados desde un archivo."""
        print(f"Cargando datos procesados desde {file_path}...")
        start_time = time.time()
        
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            self.samples = data['samples']
            self.labels = data['labels']
        
        print(f"Datos cargados en {time.time() - start_time:.2f} segundos.")
        print(f"Dataset cargado: {len(self.samples)} muestras")
    
    def save_processed_data(self, file_path):
        """Guarda los datos procesados en un archivo para uso futuro."""
        if len(self.samples) == 0:
            print("No hay datos para guardar")
            return
        
        print(f"Guardando datos procesados en {file_path}...")
        
        # Crear el directorio si no existe
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Guardar los datos
        with open(file_path, 'wb') as f:
            pickle.dump({
                'samples': self.samples,
                'labels': self.labels
            }, f)
        
        print(f"Datos guardados correctamente en {file_path}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx]), torch.tensor(self.labels[idx], dtype=torch.long)

def process_and_save_dataset(data_dir, output_path):
    """
    Procesa las imágenes ASL y guarda el dataset procesado.
    
    Args:
        data_dir (str): Directorio que contiene las imágenes ASL organizadas por clases.
        output_path (str): Ruta donde guardar el dataset procesado.
    
    Returns:
        ASLDataset: El dataset procesado.
    """
    print(f"Procesando imágenes desde {data_dir}...")
    dataset = ASLDataset(data_dir=data_dir)
    dataset.save_processed_data(output_path)
    return dataset

def load_processed_dataset(processed_data_path):
    """
    Carga un dataset ASL previamente procesado.
    
    Args:
        processed_data_path (str): Ruta al archivo de dataset procesado.
    
    Returns:
        ASLDataset: El dataset cargado.
    """
    return ASLDataset(processed_data_path=processed_data_path)

if __name__ == "__main__":
    data_dir = "./asl_alphabet_train/asl_alphabet_train"
    processed_data_dir = "./processed_data"
    processed_data_path = os.path.join(processed_data_dir, "asl_dataset_processed_2.pkl")
    os.makedirs(processed_data_dir, exist_ok=True)
    
    if os.path.exists(processed_data_path):
        print(f"Archivo de datos procesados encontrado en {processed_data_path}")
        print("Cargando datos procesados...")
        dataset = load_processed_dataset(processed_data_path)
    else:
        print("No se encontraron datos procesados.")
        print("Procesando imágenes y guardando dataset...")
        dataset = process_and_save_dataset(data_dir, processed_data_path)
    