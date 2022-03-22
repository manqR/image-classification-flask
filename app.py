from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
import numpy as np


app = Flask(__name__)

model = load_model('model.h5')
class_names = {0:'minus1', 1:'minus3', 2:'pos1'}
model.make_predict_function()

def predict_label(img_path):
    
	# i = image.load_img(img_path, target_size=(100,100))
	# i = image.img_to_array(i)/255.0
	# i = i.reshape(1, 100,100,3)
	# p = model.predict_classes(i)
	# return dic[p[0]]


    img = image.load_img(img_path, target_size=(180, 180))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    return class_names[np.argmax(score)] , 100 * np.max(score)

#routes
@app.route("/",methods=['GET','POST'])
def main():
    return render_template("index.html")

@app.route("/submit",methods=['GET','POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']

        img_path = "static/"+ img.filename
        img.save(img_path)

        p = predict_label(img_path)

        return render_template("index.html", prediction = p, img_path = img_path)

        if __name__=='__main__':
            app.run(debug = True)