import os
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
from config import *
from data_loader import data_loader
from models import model_builder, create_focal_loss, create_advanced_optimizer

class AdvancedTrainer:
    def __init__(self):
        self.models = {}
        self.histories = {}
        self.class_weights = None
        
    def train_all_models(self):
        """Train all models in the ensemble"""
        print("=== Starting Advanced Skin Disease Classification Training ===")
        
        # Load dataset
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_loader.load_dataset()
        
        # Compute class weights
        self.class_weights = data_loader.get_class_weights(y_train)
        print(f"Class weights computed: {self.class_weights}")
        
        # Create data generators
        train_gen, val_gen, steps_per_epoch, val_steps = data_loader.create_data_generators(
            X_train, y_train, X_val, y_val
        )
        
        # Build models
        print("\nBuilding models...")
        self.models = model_builder.build_ensemble_model()
        
        # Train each model
        for model_name, model in self.models.items():
            print(f"\n=== Training {model_name.upper()} ===")
            self._train_single_model(
                model, model_name, train_gen, val_gen, 
                steps_per_epoch, val_steps
            )
        
        # Evaluate on test set
        print("\n=== Final Evaluation ===")
        self._evaluate_ensemble(X_test, y_test)
        
        # Save ensemble configuration
        self._save_ensemble_config()
        
        print("\n=== Training Complete ===")
        
    def _train_single_model(self, model, model_name, train_gen, val_gen, steps_per_epoch, val_steps):
        """Train a single model with two-phase approach"""
        # Phase 1: Train only the classification head
        print("Phase 1: Training classification head...")
        self._compile_model(model, phase=1)
        
        history1 = model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=20,
            validation_data=val_gen,
            validation_steps=val_steps,
            callbacks=model_builder.get_model_callbacks(f"{model_name}_phase1"),
            class_weight=self.class_weights,
            verbose=1
        )
        
        # Phase 2: Fine-tune with unfrozen backbone
        print("Phase 2: Fine-tuning with unfrozen backbone...")
        self._unfreeze_backbone(model, model_name)
        self._compile_model(model, phase=2)
        
        history2 = model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=30,
            validation_data=val_gen,
            validation_steps=val_steps,
            callbacks=model_builder.get_model_callbacks(f"{model_name}_phase2"),
            class_weight=self.class_weights,
            verbose=1
        )
        
        # Combine histories
        combined_history = {
            'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
            'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
            'loss': history1.history['loss'] + history2.history['loss'],
            'val_loss': history1.history['val_loss'] + history2.history['val_loss']
        }
        
        self.histories[model_name] = combined_history
        
        # Plot training history
        self._plot_training_history(combined_history, model_name)
        
    def _compile_model(self, model, phase=1):
        """Compile model with appropriate settings for training phase"""
        if phase == 1:
            # Phase 1: Higher learning rate for head training
            optimizer = create_advanced_optimizer(learning_rate=1e-3)
        else:
            # Phase 2: Lower learning rate for fine-tuning
            optimizer = create_advanced_optimizer(learning_rate=1e-5)
            
        model.compile(
            optimizer=optimizer,
            loss=create_focal_loss(alpha=0.25, gamma=2.0),
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
    
    def _unfreeze_backbone(self, model, model_name):
        """Unfreeze backbone layers for fine-tuning"""
        if model_name == 'efficientnet':
            # Unfreeze last 40 layers for EfficientNet
            for layer in model.layers[-40:]:
                layer.trainable = True
        elif model_name == 'convnext':
            # Unfreeze last 30 layers for ConvNeXt
            for layer in model.layers[-30:]:
                layer.trainable = True
        elif model_name == 'vit':
            # Unfreeze last 20 layers for ViT
            for layer in model.layers[-20:]:
                layer.trainable = True
    
    def _evaluate_ensemble(self, X_test, y_test):
        """Evaluate ensemble performance on test set"""
        predictions = []
        
        print("Evaluating individual models...")
        for model_name, model in self.models.items():
            # Preprocess test images
            X_test_processed = []
            for img in X_test:
                if model_name == 'vit':
                    processed = data_loader.preprocess_image(img, target_size=(224, 224))
                else:
                    processed = data_loader.preprocess_image(img, target_size=(384, 384))
                X_test_processed.append(processed)
            
            X_test_processed = np.array(X_test_processed)
            
            # Get predictions
            pred = model.predict(X_test_processed, verbose=1)
            predictions.append(pred)
            
            # Individual model accuracy
            accuracy = np.mean(np.argmax(pred, axis=1) == y_test)
            print(f"{model_name} accuracy: {accuracy:.4f}")
        
        # Ensemble prediction (weighted average)
        ensemble_pred = (
            predictions[0] * ENSEMBLE_WEIGHTS['efficientnet'] +
            predictions[1] * ENSEMBLE_WEIGHTS['convnext'] +
            predictions[2] * ENSEMBLE_WEIGHTS['vit']
        )
        
        ensemble_accuracy = np.mean(np.argmax(ensemble_pred, axis=1) == y_test)
        print(f"Ensemble accuracy: {ensemble_accuracy:.4f}")
        
        # Detailed classification report
        y_pred_classes = np.argmax(ensemble_pred, axis=1)
        report = classification_report(
            y_test, y_pred_classes, 
            target_names=CLASSES, 
            output_dict=True
        )
        
        # Save evaluation results
        results = {
            'ensemble_accuracy': float(ensemble_accuracy),
            'individual_accuracies': {
                name: float(np.mean(np.argmax(pred, axis=1) == y_test))
                for name, pred in zip(self.models.keys(), predictions)
            },
            'classification_report': report
        }
        
        with open(os.path.join(MODELS_DIR, 'evaluation_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Plot confusion matrix
        self._plot_confusion_matrix(y_test, y_pred_classes)
        
        return ensemble_accuracy
    
    def _plot_training_history(self, history, model_name):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy
        ax1.plot(history['accuracy'], label='Training Accuracy')
        ax1.plot(history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title(f'{model_name} - Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Loss
        ax2.plot(history['loss'], label='Training Loss')
        ax2.plot(history['val_loss'], label='Validation Loss')
        ax2.set_title(f'{model_name} - Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(MODELS_DIR, f'{model_name}_training_history.png'))
        plt.close()
    
    def _plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=CLASSES,
            yticklabels=CLASSES
        )
        plt.title('Confusion Matrix - Ensemble Model')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(MODELS_DIR, 'confusion_matrix.png'))
        plt.close()
    
    def _save_ensemble_config(self):
        """Save ensemble configuration"""
        config = {
            'models': list(self.models.keys()),
            'weights': ENSEMBLE_WEIGHTS,
            'classes': CLASSES,
            'num_classes': NUM_CLASSES
        }
        
        with open(os.path.join(MODELS_DIR, 'ensemble_config.json'), 'w') as f:
            json.dump(config, f, indent=2)

def main():
    """Main training function"""
    trainer = AdvancedTrainer()
    trainer.train_all_models()

if __name__ == "__main__":
    main()