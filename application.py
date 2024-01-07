from flask import Flask, render_template, request, jsonify
import os
import base64
from PIL import Image
import tensorflow as tf
import numpy as np
from datetime import datetime

application=Flask(__name__)
app=application
# Load the saved model
loaded_model = tf.keras.models.load_model("mnist_model.h5")
# Ensure that the 'static/canvas' directory exists
static_canvas_dir = os.path.join(app.root_path, 'static', 'canvas')
os.makedirs(static_canvas_dir, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/canvas', methods=['POST'])
def save_canvas():
    try:
        canvas_data = request.form['canvas_data']
        
        # Generate a dynamic filename based on timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        file_name = f'canvas_image_{timestamp}.png'

        file_path = os.path.join(static_canvas_dir, file_name)

        # Save the canvas image
        with open(file_path, 'wb') as file:
            file.write(base64.b64decode(canvas_data.split(',')[1]))

        return jsonify({'message': 'Canvas image saved successfully!'})
    except Exception as e:
        return jsonify({'error': str(e)})
    
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json  # Use request.json to parse JSON data
        canvas_data = data['canvas_data']

        # Reshape the data
        input_data = np.array(canvas_data).reshape(-1, 28, 28, 1)

        # Make predictions using the loaded model
        predictions = loaded_model.predict(input_data)

        # Get the predicted class
        predicted_class = np.argmax(predictions[0])

        return jsonify({'predicted_class': int(predicted_class)})
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(host="0.0.0.0")
