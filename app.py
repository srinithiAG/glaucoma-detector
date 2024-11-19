from flask import Flask, render_template, request, send_from_directory
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os
from flask import render_template
############## IMPORT MODULES ##########################################################################
# Flask imports (Flask, render_template, request, send_from_directory) 
# provide the tools to create a web server, handle HTTP requests, render templates, and serve static files.

# TensorFlow (tensorflow as tf)  It is widely used for developing, training, and deploying machine learning models, 
# particularly deep learning models like neural networks

# Pillow (Image, ImageOps) provides image processing functionalities. like resizing, grayscaling, flipping, etc...

# NumPy (numpy as np) is a library used for working with arrays and matrices in Python.

# OS (os) interacts with the file system, helping with file paths and directories.



app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads') 

model = tf.keras.models.load_model('my_model2.h5')
# Load the trained model [TensorFlow/Keras model]
# This model will be used later to predict whether the uploaded image shows signs of Glaucoma

################ preprocess and predict image #######################
def import_and_predict(image_data, model):
    image = Image.open(image_data)
    # Uses the Pillow library to open the uploaded image

    image = ImageOps.fit(image, (100, 100), Image.LANCZOS)
    # Resizes the image to a 100x100 pixel dimension using the ImageOps
    # fit function. The Image.LANCZOS filter is a high-quality down-sampling algorithm

    image = image.convert('RGB')
    # Converts the image to RGB format (Red, Green, Blue)

    image_array = np.asarray(image)
    # Converts the image format to a NumPy array

    image_array = (image_array.astype(np.float32) / 255.0)
    # Normalizes the pixel values by converting them to floating-point numbers

    img_reshape = image_array[np.newaxis, ...]
    # Adds a new dimension to the array to make it compatible with the model's expected input shape.

    prediction = model.predict(img_reshape)
    # Uses the loaded model to predict whether the image shows signs of Glaucoma or not

    return prediction, image

################### upload file(img) ################
@app.route('/')
def home():
    return render_template('index.html')
# The home route renders the index.html file when the root URL (/) is accessed. 
# This will be the form where the user uploads the image

@app.route('/predict', methods=['POST'])
# define the new router. This route handles the image submission and returns the prediction result

def predict():
    if request.method == 'POST':
        file = request.files['file']
    # Checks if the request method is POST (the form was submitted)

        if file:
            filename = file.filename
            glaucoma_dir = os.path.join('Glaucomatous', filename)
            healthy_dir = os.path.join('Healthy', filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            print("Image saved at:", file_path)
        # Retrieves the filename.
        # Creates paths for possible pre-existing files in the Glaucomatous and Healthy directories (for comparison).
        # Saves the uploaded file to the UPLOAD_FOLDER (i.e., static/uploads).
        # Prints the saved file path to the console.

            if os.path.exists(glaucoma_dir):
                image_path = glaucoma_dir
            elif os.path.exists(healthy_dir):
                image_path = healthy_dir
            else:
                return '<p style="color: red; font-weight: bold;">Invalid image file selected.</p>'
            # Checks if the uploaded file already exists in the Glaucomatous or Healthy directories 
            # and assigns the appropriate path. If the file doesn't exist in either directory, it returns an error message.


############ Make prediction #############################
            prediction, image = import_and_predict(file_path, model)
            pred = prediction[0][0]
            pred_percentage = "{:.2f}".format(pred * 100)
            # Calls the import_and_predict function with the uploaded image and model.
            # prediction[0][0]: Retrieves the first prediction result.
            # pred_percentage: Converts the prediction result into a percentage with two decimal places.

            if pred > 0.5:
                result = "Your eye is Healthy. Great!!"
            else:
                result = "You are affected by Glaucoma. Please consult an ophthalmologist as soon as possible."
            # If the prediction result is greater than 0.5, the eye is predicted to be healthy. otherwise, it is classified as Glaucoma.


########## Pass the image path and prediction result to the result template #######################################
            return render_template('result.html', result=result, image_path=file_path, pred_percentage=pred_percentage, os=os)
            # return render_template('result.html', result=result)
        
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
# Defines a route for serving uploaded files from the UPLOAD_FOLDER. When accessed, it will return the specified file.

if __name__ == '__main__':
    app.run(debug=True)
# The main block ensures that the Flask app runs only when the script is executed directly (not imported as a module). 
# The debug=True flag enables debugging mode, which provides useful error messages and auto-reloads the server on code changes.