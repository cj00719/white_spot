
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

model = load_model('white_spot_cnn_model1.h5')
CLASS_NAMES = ['Healthy', 'Infected']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            img = load_img(filepath, target_size=(128, 128))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            preds = model.predict(img_array)[0]
            class_idx = np.argmax(preds)
            label = CLASS_NAMES[class_idx]
            confidence = round(float(preds[class_idx]) * 100, 2)
            return render_template('index.html', result=label, confidence=confidence, image_path=filepath)
    return render_template('index.html', result=None)

if __name__ == '__main__':
    app.run(debug=True)
