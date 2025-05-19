import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import os
import cv2
import mediapipe as mp
import pickle
from sklearn.model_selection import KFold

# Definimos las etiquetas
clases = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
          'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
          'space', 'delete', 'nothing']

# Configuramos MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Definimos la arquitectura del modelo
class ASLTranslator(nn.Module):
    def __init__(self):
        super(ASLTranslator, self).__init__()
        self.fc1 = nn.Linear(63, 130)
        self.fc2 = nn.Linear(130, 63)
        self.fc3 = nn.Linear(63, 29)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x

# Clase para cargar y procesar el dataset
class ASLDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        self.labels = []
        
        self.hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
        
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
                    except Exception as e:
                        print(f"Error procesando {img_path}: {e}")
        
        print(f"Dataset cargado: {len(self.samples)} muestras")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx]), torch.tensor(self.labels[idx], dtype=torch.long)

# Función para entrenar el modelo
def train_model(model, train_loader, val_loader, epochs=10, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        # Entrenamiento
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        
        # Validación
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        
        print(f"Epoch {epoch+1}/{epochs}, "
              f"Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss/len(val_loader):.4f}, "
              f"Val Acc: {val_acc:.2f}%")
        
        # Guardamos el mejor modelo
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "mejor_modelo_asl.pth")
            print(f"Modelo guardado con precisión de validación: {val_acc:.2f}%")
    
    # Cargamos el mejor modelo para retornarlo
    model.load_state_dict(torch.load("mejor_modelo_asl.pth"))
    return model

# Función para evaluar el modelo
def evaluate_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    
    # Para la matriz de confusión
    confusion_matrix = np.zeros((29, 29), dtype=int)
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Actualizar matriz de confusión
            for i in range(len(labels)):
                confusion_matrix[labels[i].item()][predicted[i].item()] += 1
    
    accuracy = 100 * correct / total
    print(f"Precisión del modelo en el conjunto de prueba: {accuracy:.2f}%")
    
    # Calcular métricas por clase
    for i in range(29):
        true_positives = confusion_matrix[i][i]
        false_positives = sum(confusion_matrix[:, i]) - true_positives
        false_negatives = sum(confusion_matrix[i, :]) - true_positives
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"Clase {clases[i]} - Precisión: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    
    return accuracy, confusion_matrix

# Función para entrenar el modelo en un fold específico
def train_fold(model, train_loader, val_loader, fold, epochs=10, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0

    print(f"\n--- Entrenando Fold {fold + 1} ---")
    for epoch in range(epochs):
        # Entrenamiento
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_acc = 100 * train_correct / train_total

        # Validación
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total

        print(f"Epoch {epoch+1}/{epochs}, "
              f"Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss/len(val_loader):.4f}, "
              f"Val Acc: {val_acc:.2f}%")

        # Guardamos el mejor modelo para este fold
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"mejor_modelo_fold_{fold+1}.pth")
            print(f"Modelo guardado para Fold {fold+1} con precisión de validación: {val_acc:.2f}%")

    # Cargamos el mejor modelo para este fold
    model.load_state_dict(torch.load(f"mejor_modelo_fold_{fold+1}.pth"))
    return model, best_val_acc

# Función para evaluar el modelo en un fold específico
def evaluate_fold(model, test_loader, fold):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    correct = 0
    total = 0

    # Para la matriz de confusión
    confusion_matrix = np.zeros((29, 29), dtype=int)

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Actualizar matriz de confusión
            for i in range(len(labels)):
                confusion_matrix[labels[i].item()][predicted[i].item()] += 1

    accuracy = 100 * correct / total
    print(f"\n--- Evaluación Fold {fold + 1} ---")
    print(f"Precisión del modelo en el conjunto de prueba (Fold {fold+1}): {accuracy:.2f}%")

    # Calcular métricas por clase
    print(f"Métricas por clase para Fold {fold + 1}:")
    for i in range(29):
        true_positives = confusion_matrix[i][i]
        false_positives = sum(confusion_matrix[:, i]) - true_positives
        false_negatives = sum(confusion_matrix[i, :]) - true_positives

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"Clase {clases[i]} - Precisión: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

    return accuracy, confusion_matrix

# Función para entrenar y guardar el modelo
def main(n_splits=5, epochs=20, lr=0.001):
    # Rutas para los datos
    data_dir = "./asl_alphabet_train/asl_alphabet_train"  # Directorio con las imágenes de entrenamiento
    
    # Creamos el dataset y dataloaders
    full_dataset = ASLDataset(data_dir)
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    
    # Inicializamos KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42) # Puedes cambiar random_state

    all_fold_accuracies = []
    all_fold_confusion_matrices = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1}/{n_splits}")
        print(f"{'='*50}")

        # Creamos los datasets y dataloaders para el fold actual
        train_dataset = Subset(full_dataset, train_idx)
        val_dataset = Subset(full_dataset, val_idx)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True) # Shuffle validation también es buena práctica

        # Inicializamos un nuevo modelo para cada fold
        model = ASLTranslator()

        # Entrenamos el modelo en el fold actual
        trained_model, best_val_acc = train_fold(model, train_loader, val_loader, fold, epochs=epochs, lr=lr)
        print(f"Mejor precisión de validación para Fold {fold + 1}: {best_val_acc:.2f}%")

        # Evaluamos el modelo en el conjunto de validación del fold actual (esto sirve como prueba en k-fold)
        accuracy, confusion_matrix = evaluate_fold(trained_model, val_loader, fold)
        all_fold_accuracies.append(accuracy)
        all_fold_confusion_matrices.append(confusion_matrix)

    # Dividimos el dataset en entrenamiento, validación y prueba
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    
    # Creamos el modelo
    model = ASLTranslator()
    
    # Entrenamos el modelo
    model = train_model(model, train_loader, val_loader, epochs=40)
    
    # Evaluamos el modelo
    accuracy, confusion_matrix = evaluate_model(model, test_loader)
    
    # Calculamos y mostramos el rendimiento promedio
    mean_accuracy = np.mean(all_fold_accuracies)
    print(f"\n{'='*50}")
    print(f"Rendimiento promedio de la validación cruzada ({n_splits}-fold): {mean_accuracy:.2f}%")
    print(f"{'='*50}")

    # Guardamos el modelo en formato pickle
    with open("modelo_asl.pkl", "wb") as f:
        pickle.dump(model, f)
    
    print("Modelo guardado como modelo_asl.pkl")

if __name__ == "__main__":
    main(n_splits=5, epochs=20, lr=0.001)