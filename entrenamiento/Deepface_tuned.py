import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, Conv2D, AveragePooling2D, BatchNormalization
from deepface.models.demography.Emotion import load_model
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import keras_tuner as kt

# --- 1. Definir modelo para hyperparameter tuning ---
def build_model(hp):
    input_tensor = Input(shape=(48, 48, 1))
    
    base_model = load_model()
    x = input_tensor
    for i in range(7):
        x = base_model.layers[i](x)
        base_model.layers[i].trainable = False

    for i in range(hp.Int('num_conv_layers', 1, 4)):
        filters = hp.Choice(f'filters_{i}', values=[128, 256])
        dropout_rate = hp.Choice(f'dropout_{i}', values=[0.1, 0.2, 0.25])
        x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        x = Dropout(dropout_rate)(x)

    x = BatchNormalization()(x)
    x = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.2)(x)
    output = Dense(7, activation='softmax')(x)

    model = Model(inputs=input_tensor, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=5e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# --- 2. Dataset ---
dataset_path = "Data"
img_size = (48, 48)
batch_size = 32

datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)

train_generator = datagen.flow_from_directory(
    directory=os.path.join(dataset_path, "train"),
    target_size=img_size,
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True
)

test_generator = datagen.flow_from_directory(
    directory=os.path.join(dataset_path, "test"),
    target_size=img_size,
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False
)

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weight_dict = dict(enumerate(class_weights))

# --- 3. Callbacks ---
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, min_delta=0.01, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_delta=0.01)
]

# --- 4. Keras Tuner ---
tuner = kt.Hyperband(
    build_model,
    objective='val_loss',
    max_epochs=15,
    factor=3,
    directory='deepface_finetuning',
    project_name='numerico'
)

tuner.search(
    train_generator,
    validation_data=test_generator,
    epochs=15,
    callbacks=callbacks,
    class_weight=class_weight_dict
)


# Reconstruimos el mejor modelo 
best_hps = tuner.get_best_hyperparameters(1)[0]
model = build_model(best_hps) #esto para reiniciar pesos y ahora entrenarlo por más épocas

# Entrenamos desde 0
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=30,
    class_weight=class_weight_dict,
    callbacks=callbacks
)

# Guardamos el modelo recién entrenado
model.save("archivo/ModeloDeepFaceTunedEntrenado.h5")



