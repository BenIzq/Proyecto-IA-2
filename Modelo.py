import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from CreateDataset import ASLDataset
import os
import cv2
import mediapipe as mp
import pickle
from sklearn.model_selection import KFold

clases = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
          'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
          'space', 'delete', 'nothing']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class ASLTranslator(nn.Module):
    def __init__(self):
        super(ASLTranslator, self).__init__()
        self.fc1 = nn.Linear(63, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 29)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.dropout(x)
        return x

class MetaModel(nn.Module):
    def __init__(self, input_size):
        super(MetaModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 29)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class StackingEnsemble:
    def __init__(self, base_models, meta_model):
        self.base_models = base_models
        self.meta_model = meta_model
        
    def train_meta_model(self, val_data_loader):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Recopilar predicciones de los modelos base en el conjunto de validación
        all_predictions = []
        all_labels = []
        
        for inputs, labels in val_data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Guardar etiquetas reales
            all_labels.append(labels)
            
            # Recopilar predicciones (probabilidades) de cada modelo base
            batch_predictions = []
            for model in self.base_models:
                model.eval()
                with torch.no_grad():
                    outputs = model(inputs)
                    batch_predictions.append(outputs)
            
            # Concatenar predicciones como características para el metamodelo
            stacked_predictions = torch.cat(batch_predictions, dim=1)
            all_predictions.append(stacked_predictions)
        
        # Concatenar todas las predicciones y etiquetas
        X_meta = torch.cat(all_predictions, dim=0)
        y_meta = torch.cat(all_labels, dim=0)
        
        # Entrenar el metamodelo
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.meta_model.parameters(), lr=0.001)
        
        for epoch in range(10):  # 10 epochs para el metamodelo
            self.meta_model.train()
            optimizer.zero_grad()
            
            meta_outputs = self.meta_model(X_meta)
            loss = criterion(meta_outputs, y_meta)
            
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(meta_outputs.data, 1)
            accuracy = 100 * (predicted == y_meta).sum().item() / y_meta.size(0)
            
            print(f"Meta-model Epoch {epoch+1}/10, Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")
        
        return self.meta_model
    
    def predict(self, x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = x.to(device)
        
        # Recopilar predicciones de modelos base
        base_predictions = []
        for model in self.base_models:
            model.eval()
            with torch.no_grad():
                outputs = model(x)
                base_predictions.append(outputs)
        
        # Concatenar predicciones como características para el metamodelo
        stacked_predictions = torch.cat(base_predictions, dim=1)
        
        # Obtener predicción final del metamodelo
        self.meta_model.eval()
        with torch.no_grad():
            meta_outputs = self.meta_model(stacked_predictions)
            _, predicted = torch.max(meta_outputs, 1)
        
        return predicted

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

def evaluate_ensemble(ensemble, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    correct = 0
    total = 0
    
    # Para la matriz de confusión
    confusion_matrix = np.zeros((29, 29), dtype=int)
    
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Obtener predicciones del ensemble
        predicted = ensemble.predict(inputs)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Actualizar matriz de confusión
        for i in range(len(labels)):
            confusion_matrix[labels[i].item()][predicted[i].item()] += 1
    
    accuracy = 100 * correct / total
    print(f"Precisión del ensemble en el conjunto de prueba: {accuracy:.2f}%")
    
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

def main(n_splits=5, epochs=20, lr=0.001):
    data_dir = "./asl_alphabet_train/asl_alphabet_train"  # Directorio con las imágenes de entrenamiento
    processed_data = "./processed_data"

    full_dataset = ASLDataset(processed_data_path=processed_data, file="asl_dataset_processed_2.pkl")
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))

    
    train_size = int(0.6 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    _, _, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=50)

    all_fold_accuracies = []
    all_fold_confusion_matrices = []
    all_fold_models = [] 

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
        all_fold_models.append(trained_model) 

    print("\n{:-^60}".format(" Ensemble por Stacking "))
    mean_accuracy = np.mean(all_fold_accuracies)

    # El metamodelo recibe las probabilidades de todos los modelos base
    meta_model = MetaModel(29 * len(all_fold_models))  
    stacking_ensemble = StackingEnsemble(all_fold_models, meta_model)
    
    stacking_ensemble.train_meta_model(val_loader)
    stacking_accuracy, stacking_confusion = evaluate_ensemble(stacking_ensemble, test_loader)
    stacking_base_models_state_dicts = [model.state_dict() for model in stacking_ensemble.base_models]
    stacking_dict = {
        "base_models": {f"model_{i}": state_dict for i, state_dict in enumerate(stacking_base_models_state_dicts)},
        "meta_model": stacking_ensemble.meta_model.state_dict()
    }

    print("\n{:=^60}".format(" COMPARACIÓN DE RESULTADOS "))
    print(f"Modelo único: {accuracy:.2f}%")
    print(f"Promedio K-Fold: {mean_accuracy:.2f}%")
    print(f"Ensemble por Stacking: {stacking_accuracy:.2f}%")


    torch.save(stacking_dict, "mejor_modelo_asl.pth")

if __name__ == "__main__":
    main(n_splits=5, epochs=10, lr=0.01)