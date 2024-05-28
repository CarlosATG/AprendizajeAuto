import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
import matplotlib.pyplot as plt
import numpy as np
import os

# Configuración del generador de imágenes para el preprocesamiento y aumento de datos
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normaliza los valores de píxeles entre 0 y 1
    shear_range=0.2,  # Aplica transformaciones de corte
    zoom_range=0.2,  # Aplica zoom aleatorio a las imágenes
    horizontal_flip=True,  # Invierte horizontalmente las imágenes
    validation_split=0.2)  # Reserva el 20% de los datos para validación

# Generador de datos de entrenamiento
train_generator = train_datagen.flow_from_directory(
    'entrena',  # Directorio de imágenes de entrenamiento
    target_size=(150, 150),  # Redimensiona las imágenes a 150x150 píxeles
    batch_size=32,  # Número de imágenes por lote
    class_mode='categorical',  # Modo de clasificación categórica
    subset='training')  # Subconjunto de entrenamiento

# Generador de datos de validación
validation_generator = train_datagen.flow_from_directory(
    'entrena',  # Directorio de imágenes de validación
    target_size=(150, 150),  # Redimensiona las imágenes a 150x150 píxeles
    batch_size=32,  # Número de imágenes por lote
    class_mode='categorical',  # Modo de clasificación categórica
    subset='validation')  # Subconjunto de validación

# Verificación del número de clases en los datos
num_clases = len(train_generator.class_indices)
print(f'Número de clases: {num_clases}')
print('Clases encontradas:', train_generator.class_indices)

# Construcción del modelo de red neuronal convolucional
model = Sequential([
    Input(shape=(150, 150, 3)),  # Define la forma de entrada de las imágenes
    Conv2D(32, (3, 3), activation='relu'),  # Capa convolucional con 32 filtros y función de activación ReLU
    MaxPooling2D((2, 2)),  # Capa de pooling para reducir la dimensionalidad
    Conv2D(64, (3, 3), activation='relu'),  # Segunda capa convolucional con 64 filtros
    MaxPooling2D((2, 2)),  # Segunda capa de pooling
    Conv2D(128, (3, 3), activation='relu'),  # Tercera capa convolucional con 128 filtros
    MaxPooling2D((2, 2)),  # Tercera capa de pooling
    Flatten(),  # Aplanar la entrada para la capa completamente conectada
    Dense(512, activation='relu'),  # Capa densa con 512 neuronas y activación ReLU
    Dropout(0.5),  # Capa Dropout para reducir el sobreajuste
    Dense(num_clases, activation='softmax')  # Capa de salida con activación softmax para clasificación
])

# Compilación del modelo con el optimizador Adam y la función de pérdida de entropía cruzada categórica
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenamiento del modelo
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=25)  # Número de épocas de entrenamiento

# Evaluación del modelo en los datos de validación
loss, accuracy = model.evaluate(validation_generator)
print(f'Pérdida: {loss}')
print(f'Precisión: {accuracy}')

# Guardado del modelo entrenado
ruta_modelo = os.path.join(os.path.expanduser("~"), 'modelo_clasificacion_animales.h5')
model.save(ruta_modelo)
print("El modelo ha sido guardado en:", ruta_modelo)

# Visualización de los resultados de entrenamiento
plt.plot(history.history['accuracy'], label='Precisión de entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión de validación')
plt.xlabel('Epoch')
plt.ylabel('Precisión')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Pérdida de entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida de validación')
plt.xlabel('Epoch')
plt.ylabel('Pérdida')
plt.legend()
plt.show()
