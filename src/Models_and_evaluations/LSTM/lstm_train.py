import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from joblib import dump, load
from torch.utils.tensorboard import SummaryWriter
import os
import matplotlib
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

file_path = "D:/Education/Graduation Project/url-classification-system/data/final_dataset_with_selected_features.csv"
data = pd.read_csv(file_path)

urls = data['url']
y = data['label']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    urls, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

char_to_idx = {char: idx + 1 for idx, char in enumerate(set(''.join(X_train)))}
max_length = 100   

def url_to_sequence(url):
    return [char_to_idx.get(char, 0) for char in url]

def pad_sequences(sequences, max_length):
    padded = []
    for seq in sequences:
        if len(seq) < max_length:
            padded_seq = seq + [0] * (max_length - len(seq))
        else:
            padded_seq = seq[:max_length]
        padded.append(padded_seq)
    return np.array(padded)

X_train_seq = [url_to_sequence(url) for url in X_train]
X_test_seq = [url_to_sequence(url) for url in X_test]

X_train_padded = pad_sequences(X_train_seq, max_length)
X_test_padded = pad_sequences(X_test_seq, max_length)

X_train_tensor = torch.LongTensor(X_train_padded)
X_test_tensor = torch.LongTensor(X_test_padded)

y_train_tensor = torch.LongTensor(y_train)
y_test_tensor = torch.LongTensor(y_test)

class URLDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=2, bidirectional=True, dropout=0.5):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, _) = self.lstm(embedded)
        hidden = self.dropout(hidden)
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        return self.fc(hidden)

def load_preprocess_data(file_path):
    df = pd.read_csv(file_path)

    print(f"Dataset shape: {df.shape}")
    print(f"Number of phishing URLs: {sum(df['label'] == 1)}")
    print(f"Number of legitimate URLs: {sum(df['label'] == 0)}")

    X = df.drop(['url', 'label'], axis=1)
    y = df['label']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y.values, list(X.columns)


# Parameter settings
vocab_size = len(char_to_idx) + 1
embedding_dim = 128
hidden_dim = 128
bidirectional = True
output_dim = 1
num_layers = 2
dropout = 0.5
batch_size = 32 
k_folds = 5
num_epochs = 20
early_stopping_patience = 3

results = {
    'train_losses': [],
    'val_losses': [],
    'val_accuracies': [],
    'precisions': [],
    'recalls': [],
    'Sensitivity': [],
    'f1_scores': [],
    'aucs': []
}
best_f1 = 0
best_fold_number = 0
best_model_state = None

def train_with_cross_validation(X_train, y_train, n_splits=k_folds, batch_size=batch_size, num_epochs=num_epochs, learning_rate=0.001):
    
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"Fold {fold + 1}/{k_folds}")
    
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
    
        train_dataset = URLDataset(X_tr, y_tr)
        val_dataset = URLDataset(X_val, y_val)
    
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
        model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, 
                          num_layers=num_layers, bidirectional=bidirectional, dropout=dropout).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    
        log_dir = f'Secure_Cyber_Systems/lstm_train/logs/fold_{fold}'
        writer = SummaryWriter(log_dir=log_dir)
    
        best_loss = float('inf')
        counter = 0
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
    
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            epoch_loss = 0          

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
            
                optimizer.zero_grad()
                predictions = model(batch_X).squeeze(1)
                loss = criterion(predictions, batch_y.float())
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                train_loss += loss.item()
                train_total += batch_y.size(0)
                predicted = (torch.sigmoid(predictions) > 0.5).float()
                train_correct += (predicted == batch_y.float()).sum().item()

            train_loss /= len(train_loader)
            train_accuracy = train_correct / train_total
            train_accuracies.append(train_accuracy)
            train_losses.append(train_loss)
        
            model.eval()
            correct = 0
            total = 0
            val_loss = 0
            all_predictions = []
            all_labels = []
            all_probs = []
        
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(device)
                    batch_y = batch_y.to(device)
                
                    predictions = model(batch_X).squeeze(1)
                    loss = criterion(predictions, batch_y.float())
                    val_loss += loss.item()
                    predicted = (torch.sigmoid(predictions) > 0.5).float()
                    probs = torch.sigmoid(predictions)
                
                    total += batch_y.size(0)
                    correct += (predicted == batch_y.float()).sum().item()
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(batch_y.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
        
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            accuracy = correct / total
            val_accuracies.append(accuracy)
            val_accuracy = correct / total

            val_precision = precision_score(all_labels, all_predictions)
            val_recall = recall_score(all_labels, all_predictions)
            val_sensitivity = recall_score(all_labels, all_predictions, average='binary')
            val_f1 = f1_score(all_labels, all_predictions)
            val_auc = roc_auc_score(all_labels, all_probs)

            print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {accuracy:.4f}, Train Acc: {train_accuracy:.4f}")
        
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                counter = 0
            else:
                counter += 1
                if counter >= early_stopping_patience:
                    print("Early stopping triggered")
                    break
    
        print(f"Fold {fold + 1} Results:")
        print(f"Accuracy: {val_accuracy:.4f}")
        print(f"Precision: {val_precision:.4f}")
        print(f"Recall: {val_recall:.4f}")
        print(f"Sensitivity: {val_sensitivity:.4f}")
        print(f"F1 Score: {val_f1:.4f}")
        print(f"AUC: {val_auc:.4f}")

        results['train_losses'].append(train_losses)
        results['val_losses'].append(val_losses)
        results['val_accuracies'].append(val_accuracies)
    
        precision = precision_score(all_labels, all_predictions)
        recall = recall_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions)
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        auc = roc_auc_score(all_labels, all_probs)
    
        results['precisions'].append(precision)
        results['recalls'].append(recall)
        results['f1_scores'].append(f1)
        results['aucs'].append(auc)
    
        writer.close()
    
        model_dir = f'Secure_Cyber_Systems/lstm_train/models/fold_{fold}'
        os.makedirs(model_dir, exist_ok=True)
        dump(model, f'{model_dir}/phishing_detector_lstm.joblib')
        print(f"Model for fold {fold + 1} saved as {model_dir}/phishing_detector_lstm.joblib")
    
        dump(char_to_idx, f'{model_dir}/char_to_idx.joblib')
        print(f"Character to index mapping for fold {fold + 1} saved as {model_dir}/char_to_idx.joblib")

        fold_results.append({
            'model': model,
            'accuracy': val_accuracy,
            'precision': val_precision,
            'recall': val_recall,
            'sensitivity': val_sensitivity,
            'f1': val_f1,
            'auc': val_auc,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies,
        })

    best_fold = max(range(len(fold_results)), key=lambda i: fold_results[i]['f1'])
    best_model = fold_results[best_fold]['model']

    print(f"\nBest model is from fold {best_fold}")
    print(f"Accuracy: {fold_results[best_fold]['accuracy']:.4f}")
    print(f"Precision: {fold_results[best_fold]['precision']:.4f}")
    print(f"Recall: {fold_results[best_fold]['recall']:.4f}")
    print(f"Sensitivity: {fold_results[best_fold]['sensitivity']:.4f}")
    print(f"F1 Score: {fold_results[best_fold]['f1']:.4f}")
    print(f"AUC: {fold_results[best_fold]['auc']:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(fold_results[best_fold]['train_losses'], label='Training Loss')
    plt.plot(fold_results[best_fold]['val_losses'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.savefig('loss_curves.png')
    plt.show()
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(fold_results[best_fold]['train_accuracies'], label='Training Accuracy')
    plt.plot(fold_results[best_fold]['val_accuracies'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Training and Validation Accuracy (Best Fold {best_fold + 1})')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1.0)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.savefig('accuracy_curve_best_fold.png')
    plt.show()
    plt.close()

    for fold in range(len(fold_results)):
       plt.plot(fold_results[fold]['train_losses'], label=f'Fold {fold+1} Training Loss')
       plt.plot(fold_results[fold]['val_losses'], label=f'Fold {fold+1} Validation Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Across Folds (average of all folds)')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, max(max(f['train_losses']) for f in fold_results) * 1.1)
    plt.yticks(np.arange(0, max(max(f['train_losses']) for f in fold_results) * 1.1, 0.1))
    plt.savefig('Secure_Cyber_Systems/lstm_train/loss_curve_all_folds.png')
    plt.show()

    for fold in range(len(fold_results)):
       plt.plot(fold_results[fold]['val_accuracies'], label=f'Fold {fold+1} Validation Accuracy')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy Across Folds (average of all folds)')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1.0)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.savefig('Secure_Cyber_Systems/lstm_train/accuracy_curve_all_folds.png')
    plt.show()

    return best_model, fold_results

def evaluate_model(model, X_test, y_test, batch_size=32):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    test_dataset = URLDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    all_test_preds = []
    all_test_labels = []
    all_test_probs = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            predictions = model(batch_X).squeeze(1)
            probs = torch.sigmoid(predictions)
            preds = (probs > 0.5).float()

            all_test_preds.extend(preds.cpu().numpy())
            all_test_labels.extend(batch_y.cpu().numpy())
            all_test_probs.extend(probs.cpu().numpy())

    test_accuracy = accuracy_score(all_test_labels, all_test_preds)
    test_precision = precision_score(all_test_labels, all_test_preds)
    test_recall = recall_score(all_test_labels, all_test_preds)
    test_f1 = f1_score(all_test_labels, all_test_preds)
    test_auc = roc_auc_score(all_test_labels, all_test_probs)

    print("\nFinal Evaluation on Independent Test Set")
    print(f"test dataset size: {len(all_test_labels)}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print(f"Test AUC: {test_auc:.4f}")

    cm = confusion_matrix(all_test_labels, all_test_preds)
    tn, fp, fn, tp = cm.ravel()

    print("\nConfusion Matrix:")
    print(cm)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Legitimate", "Phishing"],
            yticklabels=["Legitimate", "Phishing"])

    plt.xlabel("Predicted ")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Test Set")
    plt.savefig('confusion_matrix.png')
    plt.show()
    plt.close()

    fpr, tpr, thresholds = roc_curve(all_test_labels, all_test_probs)
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {test_auc:.4f})')
    plt.plot([0,1], [0,1], linestyle='--')  
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Test Set)')
    plt.legend()
    plt.savefig('roc_curve.png')
    plt.grid(True)
    plt.show()
    plt.close()

    plt.figure(figsize=(8,6))

    fpr, tpr, _ = roc_curve(all_test_labels, all_test_probs)
    auc_value = roc_auc_score(all_test_labels, all_test_probs)

    plt.plot(fpr, tpr, linewidth=2, label=f'Best Model ROC (AUC = {auc_value:.4f})')
    plt.plot([0, 1], [0, 1], linestyle='--')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUC_curve.png')
    plt.legend()
    plt.grid(True)

    plt.savefig('AUC_curve.png')
    plt.show()
    plt.close()

    return {
        'accuracy': test_accuracy,
        'precision': test_precision,
        'recall': test_recall,
        'f1': test_f1,
        'auc': test_auc,
        'confusion_matrix': cm
 }

def main():
    X, y, feature_names = load_preprocess_data(file_path)

    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    best_model, fold_results = train_with_cross_validation(
       X_train_tensor, y_train_tensor
   )
    test_results = evaluate_model(best_model, X_test_tensor, y_test_tensor)

    print(f"Average Precision: {np.mean([f['precision'] for f in fold_results]):.4f}")
    print(f"Average Recall: {np.mean([f['recall'] for f in fold_results]):.4f}")
    print(f"Average F1 Score: {np.mean([f['f1'] for f in fold_results]):.4f}")
    print(f"Average AUC: {np.mean([f['auc'] for f in fold_results]):.4f}")

if __name__ == "__main__":
    main()