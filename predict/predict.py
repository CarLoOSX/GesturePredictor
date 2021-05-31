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
width, height = 175, 175


def predict(file):
    x = load_img(file, target_size=(width, height))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)

    array = cnn.predict(x)
    result = array[0]

    print(result.dtype)

    print(result)

    answer = np.argmax(result)

    print(np.max(result))
    if np.max(result) > 0.9:
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
    else:
        print("ERROR")
    return answer


predict(root_dir + "/plots/LEFT_1622481685463.png")

predict(root_dir + "/plots/LEFT_1622481706561.png")

predict(root_dir + "/plots/right.png")

predict(root_dir + "/plots/TOP_1622481712506.png")

# predict(root_dir + "/plots/TOP_1622223205189.png")

# predict("/Users/carloosx/Desktop/a.png")
# predict("/Users/carloosx/Desktop/b.png")
# predict("/Users/carloosx/Desktop/c.png")


predict("/Users/carloosx/Desktop/Unknown.jpeg")
