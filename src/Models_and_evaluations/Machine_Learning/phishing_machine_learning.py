#new
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import joblib
import time
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings('ignore')

np.random.seed(42)


class PhishingURLClassifier:
    def __init__(self, csv_path):
        """Initialize the classifier with the path to the dataset"""
        self.csv_path = csv_path
        self.data = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.results = {}
        self.scaler = None

    def load_and_preprocess_data(self):
        """Load and preprocess the dataset with 70/15/15 split"""
        try:
            print("Loading dataset...")
            self.data = pd.read_csv(self.csv_path)

            print(f"Dataset shape: {self.data.shape}")
            print(f"Class distribution:\n{self.data['label'].value_counts(normalize=True)}")

            X = self.data.drop('label', axis=1)

            if 'url' in X.columns:
                X = X.drop('url', axis=1)

            y = self.data['label']

            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=0.30, random_state=42, stratify=y
            )

            self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(
                X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
            )

            self.scaler = StandardScaler()
            self.X_train = self.scaler.fit_transform(X_train)
            self.X_val   = self.scaler.transform(self.X_val)
            self.X_test  = self.scaler.transform(self.X_test)

            total = len(y)
            print(f"\nData split summary:")
            print(f"  Training set  : {len(y_train):>6} samples  ({len(y_train)/total*100:.1f}%)")
            print(f"  Validation set: {len(self.y_val):>6} samples  ({len(self.y_val)/total*100:.1f}%)")
            print(f"  Test set      : {len(self.y_test):>6} samples  ({len(self.y_test)/total*100:.1f}%)")

            self.y_train = y_train

        except Exception as e:
            print(f"Error in data loading and preprocessing: {str(e)}")
            raise

        return self

    def train_models(self):
        """Train and evaluate multiple models — select best on VALIDATION set"""
        try:
            models = {
                'Decision Tree': DecisionTreeClassifier(random_state=42),
                'Random Forest': RandomForestClassifier(random_state=42),
                'SVM': SVC(probability=True, random_state=42),
                'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42),
                'LightGBM': LGBMClassifier(random_state=42, verbose=-1)
            }

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            n_classes = len(np.unique(self.y_train))
            avg_method = 'binary' if n_classes == 2 else 'weighted'

            print("\n" + "="*65)
            print(f"{'Model':<16} {'CV F1':>8} {'Val F1':>8} {'Val Acc':>8} {'Time':>8}")
            print("="*65)

            for name, model in models.items():
                start_time = time.time()

                cv_scores = cross_val_score(
                    model, self.X_train, self.y_train, cv=cv, scoring='f1_weighted'
                )

                model.fit(self.X_train, self.y_train)

                y_val_pred = model.predict(self.X_val)
                val_accuracy  = accuracy_score(self.y_val, y_val_pred)
                val_precision = precision_score(self.y_val, y_val_pred, average=avg_method)
                val_recall    = recall_score(self.y_val, y_val_pred, average=avg_method)
                val_f1        = f1_score(self.y_val, y_val_pred, average=avg_method)

                val_auc = 0
                if n_classes == 2 and hasattr(model, "predict_proba"):
                    y_val_prob = model.predict_proba(self.X_val)[:, 1]
                    fpr, tpr, _ = roc_curve(self.y_val, y_val_prob)
                    val_auc = auc(fpr, tpr)

                y_test_pred = model.predict(self.X_test)
                test_accuracy  = accuracy_score(self.y_test, y_test_pred)
                test_precision = precision_score(self.y_test, y_test_pred, average=avg_method)
                test_recall    = recall_score(self.y_test, y_test_pred, average=avg_method)
                test_f1        = f1_score(self.y_test, y_test_pred, average=avg_method)

                test_auc = 0
                if n_classes == 2 and hasattr(model, "predict_proba"):
                    y_test_prob = model.predict_proba(self.X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(self.y_test, y_test_prob)
                    test_auc = auc(fpr, tpr)

                train_time = time.time() - start_time

                self.results[name] = {
                    'model': model,
                    'cv_f1': cv_scores.mean(),
                    'val_accuracy': val_accuracy,
                    'val_precision': val_precision,
                    'val_recall': val_recall,
                    'val_f1': val_f1,
                    'val_auc': val_auc,

                    'test_accuracy': test_accuracy,
                    'test_precision': test_precision,
                    'test_recall': test_recall,
                    'test_f1': test_f1,
                    'test_auc': test_auc,
                    'train_time': train_time
                }

                print(f"{name:<16} {cv_scores.mean():>8.4f} {val_f1:>8.4f} {val_accuracy:>8.4f} {train_time:>7.2f}s")

            print("="*65)

            self.models = {name: info['model'] for name, info in self.results.items()}

            self.best_model_name = max(self.results, key=lambda x: self.results[x]['val_f1'])
            self.best_model = self.results[self.best_model_name]['model']

            print(f"\nBest model (by validation F1): {self.best_model_name}")
            print(f"  Validation F1 : {self.results[self.best_model_name]['val_f1']:.4f}")
            print(f"  Test F1       : {self.results[self.best_model_name]['test_f1']:.4f}")

            print("\nClassification Report on TEST set (best model):")
            print(classification_report(self.y_test, self.best_model.predict(self.X_test)))

        except Exception as e:
            print(f"Error in model training: {str(e)}")
            raise

        return self

    def plot_heatmap(self):
        """Plot heatmap for validation AND test metrics"""
        try:
            val_metrics = pd.DataFrame(self.results).T[
                ['val_accuracy', 'val_precision', 'val_recall', 'val_f1', 'val_auc']
            ].apply(pd.to_numeric, errors='coerce')
            val_metrics.columns = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']

            test_metrics = pd.DataFrame(self.results).T[
                ['test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_auc']
            ].apply(pd.to_numeric, errors='coerce')
            test_metrics.columns = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']

            fig, axes = plt.subplots(1, 2, figsize=(18, 6))

            sns.heatmap(val_metrics, annot=True, fmt=".4f", cmap="YlGnBu", ax=axes[0])
            axes[0].set_title("Validation Set — Model Evaluation Metrics")

            sns.heatmap(test_metrics, annot=True, fmt=".4f", cmap="YlOrRd", ax=axes[1])
            axes[1].set_title("Test Set — Model Evaluation Metrics")

            plt.tight_layout()
            plt.savefig("model_evaluation_heatmaps.png", dpi=150, bbox_inches='tight')
            plt.show()
            print("Heatmaps saved to model_evaluation_heatmaps.png")

        except Exception as e:
            print(f"Error in plotting: {str(e)}")

        return self

    def hyperparameter_tuning(self):
        """Hyperparameter tuning — scored on VALIDATION set via CV on training data"""
        try:
            print(f"\nPerforming hyperparameter tuning for {self.best_model_name}...")

            param_grids = {
                'Decision Tree': {
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'Random Forest': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [None, 20, 30],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                },
                'SVM': {
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.1, 0.01],
                    'kernel': ['rbf', 'linear', 'poly']
                },
                'XGBoost': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7, 9],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                },
                'LightGBM': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7, 9],
                    'num_leaves': [31, 50, 70],
                    'min_child_samples': [5, 10, 20],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
            }

            param_grid = param_grids.get(self.best_model_name, {})
            if not param_grid:
                print("No parameter grid defined for this model type.")
                return self

            n_classes = len(np.unique(self.y_train))
            scoring = 'f1' if n_classes == 2 else 'f1_weighted'
            avg_method = 'binary' if n_classes == 2 else 'weighted'

            grid_search = GridSearchCV(
                estimator=self.best_model,
                param_grid=param_grid,
                cv=5,
                scoring=scoring,
                n_jobs=-1
            )

            grid_search.fit(self.X_train, self.y_train)

            best_params = grid_search.best_params_
            self.best_model = grid_search.best_estimator_

            print(f"Best parameters: {best_params}")

            y_val_pred = self.best_model.predict(self.X_val)
            tuned_val_f1 = f1_score(self.y_val, y_val_pred, average=avg_method)

            y_test_pred = self.best_model.predict(self.X_test)
            tuned_test_f1 = f1_score(self.y_test, y_test_pred, average=avg_method)

            print(f"Tuned model — Validation F1: {tuned_val_f1:.4f}")
            print(f"Tuned model — Test F1      : {tuned_test_f1:.4f}")

            self.results[self.best_model_name]['tuned_val_f1']  = tuned_val_f1
            self.results[self.best_model_name]['tuned_test_f1'] = tuned_test_f1
            self.results[self.best_model_name]['best_params']   = best_params

        except Exception as e:
            print(f"Error in hyperparameter tuning: {str(e)}")

        return self

    def save_model(self, filepath='phishing_url_classifier.joblib'):
        """Save the best model to disk"""
        try:
            model_data = {
                'model': self.best_model,
                'scaler': self.scaler,
                'model_name': self.best_model_name
            }
            joblib.dump(model_data, filepath)
            print(f"\nBest model saved to {filepath}")
        except Exception as e:
            print(f"Error saving model: {str(e)}")

        return self

    def run_full_pipeline(self):
        """Run the complete pipeline"""
        try:
            (self
             .load_and_preprocess_data()
             .train_models()
             .plot_heatmap()
             .hyperparameter_tuning()
             .save_model()
             )
        except Exception as e:
            print(f"Error in pipeline execution: {str(e)}")

        return self


if __name__ == "__main__":
    classifier = PhishingURLClassifier('/content/drive/MyDrive/colab/url-classification-system/data/final_dataset_with_selected_features.csv')
    classifier.run_full_pipeline()