import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve # Added roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from typing import Callable, Optional, Tuple

from tensorflow.keras.mixed_precision import Policy, set_global_policy

#policy = Policy('mixed_float16')
#set_global_policy(policy)


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
if len(tf.config.list_physical_devices('GPU')) > 0:
    print("GPU is available, using GPU for training.")
else:
    print("GPU is not available, using CPU for training.  Training will be slower.")

def create_custom_model(
    base_model: Callable, 
    include_top: bool = False,
    weights: str = "imagenet",
    input_shape: Tuple = (224, 224, 3),
    num_classes: int = 1,  
    dense_units: int = 256,  
    activation_dense: str = "relu",
    activation_output: str = "sigmoid", 
    trainable_base: bool = False,
    dropout_rate = 0.5,
):

    base_model = base_model(include_top=include_top, weights=weights, input_shape=input_shape)

    base_model.trainable = trainable_base 


    input_tensor = tf.keras.layers.Input(shape=input_shape)
    x = base_model(input_tensor, training=False) 

    if not include_top:  
        x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dense(dense_units, activation=activation_dense)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    output_tensor = tf.keras.layers.Dense(num_classes, activation=activation_output)(x)
    
    model = tf.keras.Model(input_tensor, output_tensor)
    return model

def create_data_generators(data_dir, image_size, batch_size, validation_split=0.15, preprocessing_function=None):
    datagen = ImageDataGenerator(
        rescale=1./255,  
        validation_split=validation_split
    )

    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary',  
        subset='training',
        shuffle=True,
        seed=13
    )
    
    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary', 
        subset='validation',
        shuffle=True,
        seed=29
    )
    return train_generator, validation_generator


def train_model(model, train_generator, validation_generator, epochs, batch_size, early_stopping_patience=10, reduce_lr_patience=5, reduce_lr_factor=0.3):

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['AUC', 'accuracy'])
    model.summary()  # Display model summary

    early_stopping = EarlyStopping(
        monitor='loss',      
        patience=early_stopping_patience, 
        restore_best_weights=True, 
        mode='min',              
        verbose=1 
    )

    reduce_lr = ReduceLROnPlateau(
        monitor = 'val_loss', 
        factor = reduce_lr_factor,
        patience = reduce_lr_patience, 
        min_lr = 1e-15,
        mode = 'min', 
        verbose = 1)
    
    history = model.fit(
        train_generator,
        steps_per_epoch=512, 
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=512, 
        callbacks=[early_stopping, reduce_lr]
    )
    return history

def evaluate_model(model, validation_generator, target_names=['Real', 'Fake']):

    true_labels = validation_generator.classes
    predictions = model.predict(validation_generator)
    predicted_labels = (predictions > 0.5).astype(int)

    print(classification_report(true_labels, predicted_labels, target_names=target_names))

    cm = confusion_matrix(true_labels, predicted_labels)
    print("Confusion Matrix:")
    print(cm)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

def plot_training_history(history):

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy vs. Epoch')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss vs. Epoch')

    plt.tight_layout()
    plt.show()

def evaluate_model_on_test_data(model, test_data_dir, preprocessing_function, image_size=(224, 224), batch_size=32):

    test_datagen = ImageDataGenerator(
        rescale=1./255
        #preprocessing_function=preprocessing_function
    )  

    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary', 
        shuffle=False 
    )

    true_labels = test_generator.classes
    predictions = model.predict(test_generator)
    predicted_labels = (predictions > 0.5).astype(int)
    print(predicted_labels)

    report = classification_report(true_labels, predicted_labels, target_names=['Real', 'Fake'], output_dict=True)
    accuracy = report['accuracy']
    precision = report['Real']['precision']
    recall = report['Real']['recall'] 
    f1_score = report['Real']['f1-score'] 

    try:
        auc = roc_auc_score(true_labels, predictions) 
    except ValueError as e:
        print(f"Error calculating AUC: {e}.  This likely means you only have one class present in the test data.")
        auc = np.nan 

    print("Classification Report:")
    print(classification_report(true_labels, predicted_labels, target_names=['Real', 'Fake']))

    cm = confusion_matrix(true_labels, predicted_labels)
    print("Confusion Matrix:")
    print(cm)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    if not np.isnan(auc):
        fpr, tpr, thresholds = roc_curve(true_labels, predictions)
        plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'auc': auc
    }
    return metrics