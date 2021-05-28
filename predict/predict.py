import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

# Prepare the model
root_dir = os.path.abspath("")

model = '../model/model.h5'
model_weights = '../model/weights.h5'
cnn = load_model(model)
cnn.load_weights(model_weights)

# Keras Properties
width, height = 250, 250


def predict(file):

    x = load_img(file, target_size=(width, height))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)

    array = cnn.predict(x)
    result = array[0]

    print(result)

    answer = np.argmax(result)

    if answer == 0:
        print("======= | Bottom | =======")
    elif answer == 1:
        print("======= | Circle | =======")
    elif answer == 2:
        print("======= | Left | =======")
    elif answer == 3:
        print("======= | Right | =======")
    elif answer == 4:
        print("======= | Top | =======")

    return answer


predict(root_dir + "/plots/BOTTOM_1622219767249.png")

predict(root_dir + "/plots/CIRCLE_1622219770135.png")


predict(root_dir + "/plots/LEFT_1622219763020.png")


predict(root_dir + "/plots/RIGHT_1622219764048.png")

predict(root_dir + "/plots/TOP_1622219766228.png")

