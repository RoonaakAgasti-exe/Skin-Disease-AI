import os
import cv2
import numpy as np
from config import *

class SkinDiseaseDataLoader:
    def __init__(self):
        self.classes = CLASSES
        self.num_classes = NUM_CLASSES
        self.train_dir = TRAIN_DIR
        self.test_dir = TEST_DIR
        
    def get_class_weights(self, y_train):
        """Compute balanced class weights for training"""
        # Simple uniform weights for demo
        return {i: 1.0 for i in range(self.num_classes)}
    
    def load_dataset(self, validation_split=0.2):
        """Load and split the dataset"""
        print("Loading dataset...")
        
        # For demo purposes, return dummy data
        X_train = np.random.rand(100, 224, 224, 3)
        y_train = np.random.randint(0, self.num_classes, 100)
        
        X_val = np.random.rand(20, 224, 224, 3)
        y_val = np.random.randint(0, self.num_classes, 20)
        
        X_test = np.random.rand(30, 224, 224, 3)
        y_test = np.random.randint(0, self.num_classes, 30)
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Number of classes: {self.num_classes}")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def get_augmentations(self):
        """Simple augmentation for demo"""
        def simple_augment(image):
            # Simple random transformations
            if np.random.random() > 0.5:
                image = cv2.flip(image, 1)  # Horizontal flip
            if np.random.random() > 0.7:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            return image
        return simple_augment
    
    def preprocess_image(self, image, target_size=(384, 384)):
        """Preprocess single image for prediction"""
        # Resize
        img = cv2.resize(image, target_size)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Standardize (approximate ImageNet normalization)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        
        return img

# Global data loader instance
data_loader = SkinDiseaseDataLoader()