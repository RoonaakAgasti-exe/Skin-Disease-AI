import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetV2M, ConvNeXtTiny
from tensorflow.keras.regularizers import l2
import numpy as np
from config import *

class AdvancedSkinDiseaseModels:
    def __init__(self):
        self.num_classes = NUM_CLASSES
        self.dropout_rate = DROPOUT_RATE
        self.weight_decay = WEIGHT_DECAY
        
    def build_efficientnet_v2(self, input_shape=(384, 384, 3), trainable_layers=0):
        """Build EfficientNetV2-M with fine-tuning capability"""
        # Load pre-trained EfficientNetV2-M
        base_model = EfficientNetV2M(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        
        # Freeze base layers initially
        base_model.trainable = False
        
        # Add custom classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(1024, activation='relu', kernel_regularizer=l2(self.weight_decay))(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(512, activation='relu', kernel_regularizer=l2(self.weight_decay))(x)
        outputs = Dense(self.num_classes, activation='softmax', name='predictions')(x)
        
        model = Model(inputs=base_model.input, outputs=outputs, name='EfficientNetV2M_Skin')
        
        return model, base_model
    
    def build_convnext_tiny(self, input_shape=(384, 384, 3), trainable_layers=0):
        """Build ConvNeXt-Tiny with advanced head"""
        # Load pre-trained ConvNeXt-Tiny
        base_model = ConvNeXtTiny(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        
        # Freeze base layers initially
        base_model.trainable = False
        
        # Add custom classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = LayerNormalization(epsilon=1e-6)(x)
        x = Dense(1024, activation='gelu', kernel_regularizer=l2(self.weight_decay))(x)
        x = LayerNormalization(epsilon=1e-6)(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(512, activation='gelu', kernel_regularizer=l2(self.weight_decay))(x)
        outputs = Dense(self.num_classes, activation='softmax', name='predictions')(x)
        
        model = Model(inputs=base_model.input, outputs=outputs, name='ConvNeXtTiny_Skin')
        
        return model, base_model
    
    def build_vision_transformer(self, input_shape=(224, 224, 3)):
        """Build custom Vision Transformer for skin disease classification"""
        from tensorflow.keras.applications import ResNet50
        
        # Use ResNet50 as backbone for ViT-like architecture
        backbone = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        backbone.trainable = False
        
        # Patch embedding
        x = backbone.output
        x = GlobalAveragePooling2D()(x)
        
        # Transformer-like attention mechanism
        attention = Dense(512, activation='relu')(x)
        attention = Dense(512, activation='softmax')(attention)
        attended = Multiply()([x, attention])
        
        # Classification head
        x = Add()([x, attended])
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(512, activation='relu', kernel_regularizer=l2(self.weight_decay))(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)
        outputs = Dense(self.num_classes, activation='softmax', name='predictions')(x)
        
        model = Model(inputs=backbone.input, outputs=outputs, name='ViT_Skin')
        
        return model, backbone
    
    def build_ensemble_model(self):
        """Build ensemble model combining all three architectures"""
        # Create individual models
        effnet_model, _ = self.build_efficientnet_v2()
        convnext_model, _ = self.build_convnext_tiny()
        vit_model, _ = self.build_vision_transformer((224, 224, 3))
        
        # Create ensemble
        models = {
            'efficientnet': effnet_model,
            'convnext': convnext_model,
            'vit': vit_model
        }
        
        return models
    
    def get_model_callbacks(self, model_name):
        """Get training callbacks for a model"""
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(MODELS_DIR, f"{model_name}_best.h5"),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            tf.keras.callbacks.CSVLogger(
                os.path.join(MODELS_DIR, f"{model_name}_training.log")
            )
        ]
        
        return callbacks

# Global model builder instance
model_builder = AdvancedSkinDiseaseModels()

def create_focal_loss(alpha=0.25, gamma=2.0):
    """Create focal loss for handling class imbalance"""
    def focal_loss(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        # Calculate focal loss
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.math.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        
        return tf.reduce_mean(tf.reduce_sum(loss, axis=1))
    
    return focal_loss

def create_advanced_optimizer(learning_rate=LEARNING_RATE, warmup_steps=1000):
    """Create advanced optimizer with warmup"""
    # Cosine decay with warmup
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=learning_rate,
        decay_steps=10000,
        alpha=0.01
    )
    
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule,
        weight_decay=WEIGHT_DECAY
    )
    
    return optimizer