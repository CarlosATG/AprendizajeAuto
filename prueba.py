import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Cargar el modelo guardado
model = tf.keras.models.load_model('modelo_clasificacion_animales.h5')

# Cargar las clases
train_datagen = image.ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(
    'entrena',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical')

clases = train_generator.class_indices
clases = {v: k for k, v in clases.items()}  # Invertir el diccionario para obtener las clases por índice

# Función para predecir la clase de una imagen
def predecir_imagen(ruta_imagen):
    img = image.load_img(ruta_imagen, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediccion = model.predict(img_array)
    clase_predicha = clases[np.argmax(prediccion)]
    return clase_predicha

# Ejemplo de uso
if __name__ == "__main__":
    ruta_imagen = 'federico.jpg'
    print(f'La imagen es clasificada como: {predecir_imagen(ruta_imagen)}')
