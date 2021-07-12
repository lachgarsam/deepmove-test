import flask
from werkzeug.utils import secure_filename
from flask import send_file, render_template
from flask import Response
import werkzeug
import os
import scipy.misc
import tensorflow
import pickle
import numpy
import cv2
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from scipy import ndimage
import pandas as pd

def unpickle_patch(file):
    patch_bin_file = open(file, 'rb')
    patch_dict = pickle.load(patch_bin_file, encoding='bytes')
    return patch_dict


app = flask.Flask("Deepmove_app")


model = keras.models.load_model('model_DeepMove.h5')




def get_nbr(im):
    #im = cv2.imread(fich)
    im2 = np.copy(im)
    plt.imshow(im2, cmap='gray')
    if len(im.shape) == 3:
        im2=im2[:,:,0]
    res = cv2.resize(im2, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
    res = res.reshape((1,28,28,1))
    k = model.predict(res).T
    k = k.tolist()
    result = "Le nombre manuscrit est " + str(k.index(max(k)))
    f = open("resultat.json", "w")
    f.write(result)
    f.close()
    return result

def Mnist_predict():
    
    global sess
    global graph
    global secure_filenam
    #Reading the image file from the path it was saved in previously.
    img = plt.imread(os.path.join(app.root_path, secure_filenam))

    
    predicted_class = get_nbr(img)
                
    return flask.render_template(template_name_or_list="prediction_result.html", predicted_class=get_nbr(img))
        
    

app.add_url_rule(rule="/predict/", endpoint="predict", view_func=Mnist_predict)

def upload_image():
   
    
    global secure_filenam
    if flask.request.method == "POST":
        img_file = flask.request.files["image_file"]
        secure_filenam = secure_filename(img_file.filename)
        img_path = os.path.join(app.root_path, secure_filenam)
        img_file.save(img_path)
        print("Image uploaded successfully.")
        return flask.redirect(flask.url_for(endpoint="predict"))
    return "Image upload failed."


app.add_url_rule(rule="/upload/", endpoint="upload", view_func=upload_image, methods=["POST"])

def redirect_upload():
    
    
    return flask.render_template(template_name_or_list="upload_image.html")

app.add_url_rule(rule="/", endpoint="homepage", view_func=redirect_upload)


@app.route('/return-files/')
def return_file():
	try:
		return send_file('resultat.json', attachment_filename='resultat.json')
	except Exception as e:
		return str(e)





if __name__ == '__main__':
    port = int(os.environ.get('SERVER_PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)