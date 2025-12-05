"""
Model CNN untuk Deteksi Ekspresi Wajah
Mendukung Custom CNN dan Transfer Learning (MobileNetV2)
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dropout, Flatten, 
    Dense, BatchNormalization, GlobalAveragePooling2D, Input, Rescaling, Concatenate
)
from tensorflow.keras.applications import MobileNetV2


def create_expression_model(input_shape=(48, 48, 1), num_classes=7):
    """
    Membuat model CNN custom (Simple Baseline)
    """
    model = Sequential(name='Expression_CNN')
    
    # Blok Konvolusi 1
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', 
                     input_shape=input_shape, name='conv1'))
    model.add(BatchNormalization(name='bn1'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2'))
    model.add(BatchNormalization(name='bn2'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='pool1'))
    model.add(Dropout(0.25, name='dropout1'))
    
    # Blok Konvolusi 2
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='conv3'))
    model.add(BatchNormalization(name='bn3'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='conv4'))
    model.add(BatchNormalization(name='bn4'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='pool2'))
    model.add(Dropout(0.25, name='dropout2'))
    
    # Blok Konvolusi 3
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='conv5'))
    model.add(BatchNormalization(name='bn5'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='conv6'))
    model.add(BatchNormalization(name='bn6'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='pool3'))
    model.add(Dropout(0.25, name='dropout3'))
    
    # Bagian Klasifikasi
    model.add(Flatten(name='flatten'))
    model.add(Dense(256, activation='relu', name='fc1'))
    model.add(BatchNormalization(name='bn8'))
    model.add(Dropout(0.5, name='dropout5'))
    
    model.add(Dense(128, activation='relu', name='fc2'))
    model.add(BatchNormalization(name='bn9'))
    model.add(Dropout(0.5, name='dropout6'))
    
    # Output Layer
    model.add(Dense(num_classes, activation='softmax', name='output'))
    
    return model


def create_transfer_model(input_shape=(48, 48, 1), num_classes=7):
    """
    Membuat model menggunakan Transfer Learning (MobileNetV2)
    Lebih robust dan akurat untuk penggunaan produksi.
    """
    inputs = Input(shape=input_shape)
    
    # 1. Convert Grayscale (1 channel) to RGB (3 channels)
    # MobileNetV2 expects 3 channels. We repeat the grayscale channel 3 times.
    x = Concatenate(axis=-1)([inputs, inputs, inputs])
    
    # 2. Preprocessing (MobileNetV2 expects [-1, 1] range)
    # Input is [0, 1] from ImageDataGenerator(rescale=1./255)
    # Rescaling to [-1, 1]: (x * 2) - 1
    x = Rescaling(scale=2.0, offset=-1.0)(x)
    
    # 3. Base Model (MobileNetV2)
    # include_top=False means we remove the final classification layers
    # weights='imagenet' uses pre-trained weights
    base_model = MobileNetV2(
        input_shape=(48, 48, 3), # MobileNetV2 minimum is 32x32
        include_top=False, 
        weights='imagenet'
    )
    
    # Freeze base model layers initially (optional, but good for stability)
    # base_model.trainable = False 
    # Note: For this specific task with enough data, fine-tuning immediately 
    # often works well if learning rate is low. We'll keep it trainable but use low LR.
    base_model.trainable = True
    
    x = base_model(x)
    
    # 4. Custom Top Layers
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs, name='MobileNetV2_Expression')
    return model


def compile_model(model, learning_rate=0.001):
    """
    Kompilasi model dengan optimizer dan loss function
    """
    from tensorflow.keras.optimizers import Adam
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=learning_rate),
        metrics=['accuracy']
    )
    
    return model


if __name__ == '__main__':
    # Test membuat model
    print("=" * 70)
    print("TESTING MODEL CREATION")
    print("=" * 70)
    
    # 1. Test Custom CNN
    print("\n[1] Custom CNN:")
    model_cnn = create_expression_model(input_shape=(48, 48, 1), num_classes=7)
    print(f"    Total Parameter: {model_cnn.count_params():,}")
    
    # 2. Test MobileNetV2
    print("\n[2] MobileNetV2 (Transfer Learning):")
    try:
        model_tl = create_transfer_model(input_shape=(48, 48, 1), num_classes=7)
        print(f"    Total Parameter: {model_tl.count_params():,}")
        print("    ✓ Berhasil dibuat")
    except Exception as e:
        print(f"    ✗ Gagal: {e}")
