# backend/app.py
from flask import Flask,render_template,request,flash
import tensorflow as tf
import os
import tensorflow_addons as tfa


import pickle


def load_and_prep_image(filename, img_shape=[480,460], scale=True):
  """
  Reads in an image from filename, turns it into a tensor and reshapes into
  (224, 224, 3).

  Parameters
  ----------
  filename (str): string filename of target image
  img_shape (int): size to resize target image to, default 224
  scale (bool): whether to scale pixel values to range(0, 1), default True
  """
  # Read in the image
  img = tf.io.read_file(filename)
  # Decode it into a tensor
  img = tf.io.decode_image(img)
  # Resize the image
  img = tf.image.resize(img, [img_shape[0], img_shape[1]])
  if scale:
    # Rescale the image (get all values between 0 and 1)
    return img/255.
  else:
    return img
app = Flask(__name__)

loaded_model = tf.keras.models.load_model("new_aug_model.h5")

labels=['Safe driving','Texting - right','Talking on the phone - right', 'Texting - left',
             'Talking on the phone - left', 'Operating the radio', 'Drinking', 'Reaching behind', 'Hair and makeup', 'Talking to passenger']
@app.route('/')
def index():
    return render_template("form.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image uploaded", 400
    
    #saving the image
    img = request.files['image']
    img_path = 'static/temp.jpg'
    img.save(img_path)

    random_img = load_and_prep_image('static/temp.jpg', scale=False) #
    pred_prob = loaded_model.predict(tf.expand_dims(random_img, axis=0),verbose=0)
    pred_class = labels[pred_prob.argmax()] # find the predicted class 
    print(pred_class)                           

    #Make prediction
    #class_label = loaded_model.predict(img)
    #activity = get_activity(class_label)
    #f"Predicted activity: {pred_class}"
    return render_template("result.html",prediction=pred_class,image_activity=img.filename) 

app.run(debug=True)
