import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import efficientnet.tfkeras as efn
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow_addons.optimizers import RectifiedAdam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

BATCH_SIZE, EPOCHS = 10, 5
generator_batch = 32
PATH = '/content/gp_data'

np.random.seed(777)
tf.random.set_seed(777)

def display_sample_images(data, data_dir, num_samples=10):
    sample_data = data.sample(num_samples)
    filepaths = sample_data['filename'].tolist()
    labels = sample_data['class'].tolist()

    fig, axes = plt.subplots(1, num_samples, figsize=(15, 15))
    for i, (filepath, label) in enumerate(zip(filepaths, labels)):
        img_path = os.path.join(data_dir, filepath)
        image = Image.open(img_path)
        axes[i].imshow(image)
        axes[i].set_title(label)
        axes[i].axis("off")
    plt.show()

def create_dataframe_from_directory(data_dir):
    csv_file = 'data.csv'
    if os.path.exists(csv_file):
        data = pd.read_csv(csv_file)
    else:
        # Get the filenames and class labels for each image
        filenames = []
        labels = []
        for class_name in os.listdir(data_dir):
            if class_name == '.DS_Store': continue
            class_dir = os.path.join(data_dir, class_name)
            for filename in os.listdir(class_dir):
                if filename == '.DS_Store': continue
                filenames.append(os.path.join(class_name, filename))
                labels.append(class_name)

        # Create a DataFrame from the filenames and class labels
        data = pd.DataFrame({'filename': filenames, 'class': labels})

        # Save the DataFrame as a CSV file
        data.to_csv(csv_file, index=False)

    return data

def create_generators(data, batch_size=generator_batch):

    # Split the data
    train_data, tmp_data = train_test_split(data, test_size=0.2, random_state=42)
    test_data, val_data = train_test_split(tmp_data, test_size=0.5, random_state=42)

    datagen = ImageDataGenerator(
        rescale=1/255.0,
        featurewise_center=True,
        featurewise_std_normalization=True,
    )
    
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    datagen.mean = imagenet_mean
    datagen.std = imagenet_std
    
    train_generator = datagen.flow_from_dataframe(
        train_data,
        directory=PATH,
        x_col='filename',
        y_col='class',
        class_mode='binary',
        batch_size=batch_size,
        seed=42,
        target_size=(244, 244)
    )
    val_generator = datagen.flow_from_dataframe(
        val_data,
        directory=PATH,
        x_col='filename',
        y_col='class',
        class_mode='binary',
        batch_size=batch_size,
        seed=42,
        target_size=(244, 244)
    )
    test_generator = datagen.flow_from_dataframe(
        test_data,
        directory=PATH,
        x_col='filename',
        y_col='class',
        class_mode='binary',
        batch_size=batch_size,
        seed=42,
        target_size=(244, 244)
    )

    return train_generator, val_generator, test_generator

def create_model(input_shape=(244, 244, 3)):
    
    base_model = efn.EfficientNetB4(include_top=False, weights='imagenet', input_shape=input_shape)
    
    # Add a classification head
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs=base_model.input, outputs=x)
    
    return model

def train_model(model, train_generator, val_generator, epochs=EPOCHS, batch_size=BATCH_SIZE):
    
    # Use Adam optimizer with the given initial learning rate and weight decay
    optimizer = RectifiedAdam(learning_rate=0.001, weight_decay=0.0005)

    # Use AUC as a metric
    auc = tf.keras.metrics.AUC(name='auc')

    # Compile the model
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', auc])

    # Set learning rate scheduler
    lr_scheduler = ReduceLROnPlateau(factor=0.25, patience=2, verbose=1)

    # Set early stopping
    early_stopping = EarlyStopping(patience=2, verbose=1)

    # Set model checkpoint
    model_checkpoint = ModelCheckpoint(
        filepath="/content/drive/MyDrive/trained_models/model_epoch_{epoch:02d}.h5",
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_best_only=False,
        verbose=1
    )

    # Set CSV Logger
    csv_logger = CSVLogger('/content/drive/MyDrive/trained_models/training_log.csv', append=True)

    # Train the model
    history = model.fit(
        train_generator,
        epochs=epochs,
        steps_per_epoch=len(train_generator),
        validation_data=val_generator,
        validation_steps=len(val_generator),
        shuffle=True,
        verbose=1,
        callbacks=[lr_scheduler, early_stopping, model_checkpoint, csv_logger]
    )

    return history

def evaluate_model(model, test_generator, batch_size=BATCH_SIZE):
    test_loss, test_accuracy, test_auc = model.evaluate(test_generator, steps=len(test_generator), batch_size=batch_size)
    print(f"Test loss: {test_loss}, Test accuracy: {test_accuracy}, Test AUC: {test_auc}")

def save_model(model, file_name):
    model.save(file_name)

def main():
    
    # Create dataframe
    data_frame = create_dataframe_from_directory(PATH)

    # Display images
    display_sample_images(data_frame, PATH)
    
    # Create generators
    train_generator, val_generator, test_generator = create_generators(data_frame)

    # Load model and weights
    model = create_model()
    
    # Train the model
    history = train_model(model, train_generator, val_generator)

    # Evaluate the model
    evaluate_model(model, test_generator)

    # Save the model
    save_model(model, "/content/drive/MyDrive/trained_models/efficientnetb4_model.h5")

if __name__ == "__main__":
    main()
