import os
import tensorflow as tf
from tensorflow.keras import backend as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
import tensorflow.keras.optimizers as optimizers

# system environment
# to remove other warning i have to compile tensorflow on my machine
# https://www.tensorflow.org/install/source
# https://stackoverflow.com/questions/64533790/streamexecutor-device-0-host-default-version-in-tensorflow
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

# Clear the session to create a new model
keras.clear_session()

# Get Directories
root_dir = os.path.abspath("")
training_data = os.path.join(root_dir, "data/training")
validation_data = os.path.join(root_dir, "data/validation")
model_folder = os.path.join(root_dir, "../model")

# Algorithm HyperParameters
epochs = 100  # number of time the training data passes throwugh the neuronal network
batch_size = 21  # numero de lotes de datos por iteracion o epoch
learning_rate = 0.0004  # escalar para multiplicar por el vector de gradiente
learning_rate_decay = 0  # una tasa que se utiliza para disminuir el learning_rate a medida que van pasando las epochs
momentum = 0.9  # Con adam no se puede usar NAdam Ejemplo de la bola de billar sobre una pendiente, no hacer saltos
# no pensar que se ha encontrado el minimo global

# Neuronal Network Structure HyperParameter
width, height = 150, 150

pool_size = (2, 2)
classes = 5
filters = (5, 5)
filters2 = (3, 3)
other_filters = (2, 2)

# Prepare the images

training_data_gen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

validation_data_gen = ImageDataGenerator(rescale=1. / 255)

training_generator = training_data_gen.flow_from_directory(
    training_data,
    target_size=(height, width),
    batch_size=batch_size,
    class_mode='categorical')
# color_mode='grayscale')

validation_generator = validation_data_gen.flow_from_directory(
    validation_data,
    target_size=(height, width),
    batch_size=batch_size,
    class_mode='categorical')
# color_mode='grayscale')

# check image shape
# for batch in training_generator:
#    print(batch[0].shape)


train_steps = training_generator.n / training_generator.batch_size
validation_steps = validation_generator.n / validation_generator.batch_size

model = Sequential()

model.add(Convolution2D(64, (5, 5), padding="same", input_shape=(width, height, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, (5, 5), padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(256, (3, 3), padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(256, (3, 3), padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dropout(0.5))

model.add(Dense(512, activation='relu'))
model.add(Dense(classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(learning_rate, decay=1e-6),
              metrics=['accuracy'])

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25)

model.fit(
    training_generator,
    steps_per_epoch=train_steps,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=[early_stop])

# Generate the model
if not os.path.exists(model_folder):
    os.mkdir(model_folder)
model.save(model_folder + '/model.h5')
model.save_weights(model_folder + '/weights.h5')

# print(model.evaluate_generator(validation_generator.n, validation_generator.batch_size))

model.summary()
