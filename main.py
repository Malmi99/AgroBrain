import tensorflow as tf
import numpy as np
import pickle
import random

# predicted label and the probability of the predicted label for the input image
def prediction_probability_label(model, img_path, class_labels, is_rgb=True):
  if is_rgb:
    img = tf.keras.utils.load_img(
                img_path, color_mode='rgb', target_size=[255, 255],
                interpolation='nearest'
            )
  else:
            img = tf.keras.utils.load_img(
                img_path, color_mode='grayscale', target_size=[255, 255],
                interpolation='nearest'
            )

  input_arr = tf.keras.preprocessing.image.img_to_array(img)
  input_arr = np.array([input_arr])  # Convert single image to a batch.
  input_arr = input_arr / 255
  pred_probs = model.predict(input_arr)[0]

  pred_class = np.argmax(pred_probs)
  pred_label = class_labels[pred_class]
  pred_prob = round(pred_probs[pred_class]*100, 2)


  return pred_label, pred_prob

newmodel = tf.keras.models.load_model('model/soil.h5')
class_labels = ['Neutral', 'Slightly Acidic', 'Slightly Alkaline' , 'Strongly Acidic','Strongly Alkaline']

import pyrebase
import requests

config = {
    "apiKey": "AIzaSyCMKCRaLueL649R4dCyek4cI_wgy8nvvZY",
    "authDomain": "crop-recommendation-b0802.firebaseapp.com",
    "databaseURL": "https://crop-recommendation-b0802-default-rtdb.firebaseio.com",
    "projectId": "crop-recommendation-b0802",
    "storageBucket": "crop-recommendation-b0802.appspot.com",
    "messagingSenderId": "343276248922",
    "appId": "1:343276248922:web:276519f5e768bc465f9d63",
    "measurementId": "G-1DQPSLVG83"
}

while True:
    firebase = pyrebase.initialize_app(config)
    db = firebase.database()
    fbdic = db.child("cropdata").get().val()

    if fbdic["imageurl"] != "None":

        img_path = "cropimage.jpg"
        response = requests.get(fbdic["imageurl"])

        with open(img_path, "wb") as f:
            f.write(response.content)

        classification, precentage = prediction_probability_label(newmodel, img_path, class_labels)
        print(f"{classification} {precentage} %")

        if precentage < 90:
            results = {"classification": "Invalid image Please try again.......", "crop": "", "fertilizer": ""}
            db.child("results").set(results)
            db.child("cropdata").update({"imageurl": "None"})
        else:
            dicclf = {'Neutral': 0, 'Slightly Acidic': 1, 'Slightly Alkaline': 2, 'Strongly Acidic': 3,
                      'Strongly Alkaline': 4}
            userInput = [fbdic["N"], fbdic["P"], fbdic["K"], fbdic["Temparature"], fbdic["Humidity"],
                         dicclf[classification], fbdic["Rainfall"]]
            pickled_model = pickle.load(open('model/crop1.pkl', 'rb'))
            result = pickled_model.predict([userInput])[0]
            print("The input provided is classified as:", result)
            results = {"classification": classification, "crop": result.split(" ")[0],
                       "fertilizer": result.split(" ")[1:]}

            db.child("results").set(results)
            db.child("cropdata").update({"imageurl": "None"})

    else:
        print("Please Upload new data waitin for response............................")
