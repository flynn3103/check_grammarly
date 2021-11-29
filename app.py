from pyvi import ViTokenizer
import numpy as np
import os
import torch
from flask import Flask, request,render_template,jsonify
from flask_cors import CORS
import torch.nn as nn
import flask
app = Flask(__name__)
CORS(app)


detector_path = '/content/Detector967.pkl'
detector_tokenizer_path = '/content/spm_tokenizer.model'




@app.route('/predict',methods=['POST'])
def predict():
    return jsonify({"intent": label[pred_label]})





if __name__=="__main__":
    port=int(os.environ.get('PORT',5000))
    app.run(port=port,debug=True,use_reloader=False)