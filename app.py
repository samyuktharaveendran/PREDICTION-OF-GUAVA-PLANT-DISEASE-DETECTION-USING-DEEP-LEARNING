from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

app = Flask(__name__)

# Define the path to the uploads folder
UPLOAD_FOLDER = './static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
model = load_model('guava_disease_model.h5')

# Load the label encoder classes
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('classes.npy', allow_pickle=True)

# Define remedies for each disease
remedies = {
    'Anthracnose': "Remove and destroy infected leaves."
                   "Apply fungicides containing copper or chlorothalonil."
                   "Ensure good air circulation and avoid overhead watering.",
    'Guava Rust': "Prune affected branches to improve air circulation."
                  "Apply fungicides to control the spread."
                  "Maintain proper plant nutrition."
}

def prepare_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0) / 255.0
    return img

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the file is in the request
    if 'file' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['file']
    
    # Ensure the upload folder exists
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    # Save and process the image
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(image_path)
    
    # Prepare image for prediction
    image = prepare_image(image_path)
    
    # Predict the class
    prediction = model.predict(image)
    predicted_class = label_encoder.inverse_transform(np.argmax(prediction, axis=1))
    
    # Get disease name
    disease_name = predicted_class[0]
    
    # Get remedy for the predicted disease, only if it's a disease
    if disease_name in remedies:
        remedy = remedies[disease_name]
    else:
        remedy = "No remedies available for this disease."
    
    # Render result with prediction and uploaded image
    return render_template('result.html', prediction=disease_name, image_path=image_path, remedy=remedy)

if __name__ == '__main__':
    app.run(debug=True)
    
    
import os

# Construct the absolute path
model_path = os.path.join(os.getcwd(), 'model', 'guava_disease_dual_model.h5')
model = tf.keras.models.load_model(model_path)

