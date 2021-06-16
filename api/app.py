from flask import Flask
from flask_restplus import Api, Resource
import pandas as pd
import json
from flask import request
from flask import jsonify
import os
import numpy as np
import uuid
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

root_dir = os.path.abspath('')
dir_temp = os.path.join(root_dir, 'temp')
model_dir = os.path.join(root_dir, 'model')
model_h5 = os.path.join(model_dir, 'model.h5')
model_weights = os.path.join(model_dir, 'weights.h5')

model = load_model(model_h5)
model.load_weights(model_weights)

# Keras Properties
width, height = 150, 150

flask_app = Flask(__name__)
app = Api(app=flask_app)

name_space = app.namespace('gesture-predictor', description='Gesture Predictor API')


@name_space.route("/")
class MainClass(Resource):
    @staticmethod
    def post():
        json_request = request.get_json(force=True)
        data = pd.DataFrame(eval(json.dumps(json_request)))
        plot = data.plot("timestamp")
        fig = plot.get_figure()
        if not os.path.exists(dir_temp):
            os.mkdir(dir_temp)
        file = dir_temp + "/" + str(uuid.uuid4()) + ".png"
        fig.savefig(file)
        result = predict(file)
        os.remove(file)
        return jsonify({"result": result})


def predict(file):
    x = load_img(file, target_size=(width, height))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)

    array = model.predict(x)
    result = array[0]

    answer = np.argmax(result)

    value = ""
    if np.max(result) > 0.8:
        if answer == 0:
            value = "MOVE_DOWN"
        elif answer == 1:
            value = "CIRCLE"
        elif answer == 2:
            value = "MOVE_LEFT"
        elif answer == 3:
            value = "MOVE_RIGHT"
        elif answer == 4:
            value = "MOVE_UP"
    else:
        value = "ERROR"
    return value
