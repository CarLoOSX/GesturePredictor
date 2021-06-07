import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

# Prepare the model
root_dir = os.path.abspath("") + "/plots/"
print(root_dir)
model = '../model/model.h5'
model_weights = '../model/weights.h5'
cnn = load_model(model)
cnn.load_weights(model_weights)

# Keras Properties
width, height = 150, 150


def predict(file):
    x = load_img(file, target_size=(width, height))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)

    array = cnn.predict(x)
    result = array[0]

    answer = np.argmax(result)

    if np.max(result) > 0.8:
        if answer == 0:
            print(file + "=> Bottom")
        elif answer == 1:
            print(file + "=> Circle")
        elif answer == 2:
            print(file + "=> Left")
        elif answer == 3:
            print(file + "=> Right")
        elif answer == 4:
            print(file + "=> Top")
    else:
        print("ERROR")
    return answer


for filename in os.listdir(root_dir):
    if filename.endswith('.png'):
        predict(os.path.join(root_dir, filename))
