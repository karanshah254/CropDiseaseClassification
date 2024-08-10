import io
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

app = Flask(__name__)

# Load the trained model
model = load_model('D:\\hackout 2024\\frontend-backend\\cnn_model_3.h5') # will have to give absolute path to model


# Define the class labels (replace with your actual class names)
class_labels = ['Aphids', 'Armyworm', 'Bacterial Blight', 'Healthy', 'Powdery Mildew', 'Target Spot']

# Function to process and predict the image
def prepare_image(image, target_size=(224, 224)):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image / 255.0

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            print("No image key in request.files")
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['image']
        if file.filename == '':
            print("No file selected")
            return jsonify({'error': 'No file selected'}), 400

        # Convert the file to a format that load_img can process
        image = load_img(io.BytesIO(file.read()))  # Updated line
        
        processed_image = prepare_image(image)

        # Debugging: Print the shape of processed image
        print(f"Processed image shape: {processed_image.shape}")

        # Check the model's input shape
        print(f"Model input shape: {model.input_shape}")

        prediction = model.predict(processed_image)

        # Debugging: Print the raw predictions
        print(f"Model prediction: {prediction}")

        predicted_index = np.argmax(prediction, axis=1)[0]

        # Debugging: Print the predicted index
        print(f"Predicted index: {predicted_index}")

        predicted_class = class_labels[predicted_index]

        if predicted_class == 'Healthy':
            return jsonify({'prediction': f'The crop is {predicted_class}'})
        else:
            return jsonify({'prediction': f'The crop is detected with the disease {predicted_class}'})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': 'An error occurred during prediction'}), 500

if __name__ == '__main__':
    app.run(debug=True)
