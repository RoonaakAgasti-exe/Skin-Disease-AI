import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
from tqdm import tqdm
from config import *
from data_loader import data_loader

class SimpleSkinDiseaseTrainer:
    def __init__(self):
        self.models = {}
        self.feature_extractors = {}
        
    def extract_features(self, image):
        """Extract simple but effective features from image"""
        # Resize to smaller size for faster processing
        img = cv2.resize(image, (64, 64))
        
        # Convert to different color spaces
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        
        # Extract color histograms
        hist_rgb = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist_hsv = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        hist_lab = cv2.calcHist([lab], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        
        # Flatten histograms
        features = np.concatenate([
            hist_rgb.flatten(),
            hist_hsv.flatten(),
            hist_lab.flatten()
        ])
        
        # Add texture features (simple approach)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
        features = np.append(features, edge_density)
        
        return features
    
    def load_and_preprocess_data(self):
        """Load and preprocess the dataset"""
        print("Loading dataset from archive...")
        
        X = []
        y = []
        
        # Load training data
        print("Loading training images...")
        for class_idx, class_name in enumerate(CLASSES):
            class_dir = os.path.join(TRAIN_DIR, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Directory {class_dir} not found")
                continue
                
            print(f"Loading {class_name}...")
            image_files = [f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # Limit samples per class for faster training
            sample_limit = min(200, len(image_files))  # Process first 200 images per class
            image_files = image_files[:sample_limit]
            
            for filename in tqdm(image_files, desc=f"Processing {class_name}"):
                try:
                    img_path = os.path.join(class_dir, filename)
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        features = self.extract_features(img)
                        X.append(features)
                        y.append(class_idx)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"Dataset loaded: {len(X)} samples, {len(np.unique(y))} classes")
        print(f"Feature dimension: {X.shape[1]}")
        
        return X, y
    
    def train_ensemble(self, X, y):
        """Train ensemble of Random Forest models"""
        print("Training ensemble model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train multiple Random Forest models with different parameters
        rf_models = []
        
        # Model 1: Standard RF
        print("Training Random Forest Model 1...")
        rf1 = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        rf1.fit(X_train, y_train)
        rf_models.append(('rf1', rf1))
        
        # Model 2: Deep trees
        print("Training Random Forest Model 2...")
        rf2 = RandomForestClassifier(
            n_estimators=150,
            max_depth=30,
            min_samples_split=3,
            random_state=123,
            n_jobs=-1
        )
        rf2.fit(X_train, y_train)
        rf_models.append(('rf2', rf2))
        
        # Model 3: Shallow trees (for diversity)
        print("Training Random Forest Model 3...")
        rf3 = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            random_state=456,
            n_jobs=-1
        )
        rf3.fit(X_train, y_train)
        rf_models.append(('rf3', rf3))
        
        # Evaluate individual models
        print("\nIndividual Model Performance:")
        for name, model in rf_models:
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"{name}: {accuracy:.4f}")
        
        # Create ensemble
        self.models = rf_models
        
        # Evaluate ensemble
        self.evaluate_ensemble(X_test, y_test)
        
        # Save models
        self.save_models()
        
        return X_test, y_test
    
    def evaluate_ensemble(self, X_test, y_test):
        """Evaluate ensemble performance"""
        print("\nEnsemble Performance:")
        
        # Get predictions from all models
        predictions = []
        for name, model in self.models:
            pred = model.predict_proba(X_test)
            predictions.append(pred)
        
        # Ensemble prediction (average probabilities)
        ensemble_pred_proba = np.mean(predictions, axis=0)
        ensemble_pred = np.argmax(ensemble_pred_proba, axis=1)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, ensemble_pred)
        print(f"Ensemble Accuracy: {accuracy:.4f}")
        
        # Detailed classification report
        print("\nClassification Report:")
        report = classification_report(
            y_test, ensemble_pred, 
            target_names=CLASSES,
            output_dict=True
        )
        
        # Print top accuracy per class
        for class_name in CLASSES:
            class_idx = CLASSES.index(class_name)
            if class_name in report:
                precision = report[class_name]['precision']
                recall = report[class_name]['recall']
                f1 = report[class_name]['f1-score']
                print(f"{class_name:20} P:{precision:.3f} R:{recall:.3f} F1:{f1:.3f}")
        
        return accuracy
    
    def save_models(self):
        """Save trained models"""
        print("Saving models...")
        model_path = os.path.join(MODELS_DIR, 'skin_disease_ensemble')
        os.makedirs(model_path, exist_ok=True)
        
        for name, model in self.models:
            model_file = os.path.join(model_path, f'{name}.pkl')
            joblib.dump(model, model_file)
        
        print(f"Models saved to {model_path}")
    
    def load_models(self):
        """Load trained models"""
        model_path = os.path.join(MODELS_DIR, 'skin_disease_ensemble')
        if not os.path.exists(model_path):
            return False
        
        self.models = []
        for name in ['rf1', 'rf2', 'rf3']:
            model_file = os.path.join(model_path, f'{name}.pkl')
            if os.path.exists(model_file):
                model = joblib.load(model_file)
                self.models.append((name, model))
        
        return len(self.models) > 0

def main():
    """Main training function"""
    print("=== Skin Disease Classification Training ===")
    print(f"Using dataset from: {TRAIN_DIR}")
    print(f"Number of classes: {NUM_CLASSES}")
    print(f"Classes: {CLASSES}")
    
    trainer = SimpleSkinDiseaseTrainer()
    
    # Load data
    X, y = trainer.load_and_preprocess_data()
    
    # Train models
    X_test, y_test = trainer.train_ensemble(X, y)
    
    print("\n=== Training Complete ===")
    print("Models are ready for inference!")

if __name__ == "__main__":
    main()