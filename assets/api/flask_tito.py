from flask import Flask
from flask import jsonify
from flask import request
import json

from functions import handle_zip, train, analyze

# HOST AND PORT
HOST = '127.0.0.1'
PORT = 5000

app = Flask(__name__)

@app.route("/zip_file", methods = ["POST"])
def get_zip_file():
    """
    Port for Handling the ZIP File
    """
    print("request recieved: getting the data")
    
    data = request.get_data().decode()
    resp = jsonify(handle_zip(data))
    resp.headers.add("Access-Control-Allow-Origin", "*")
    return resp

@app.route("/train", methods = ["POST"])
def train_it():
    """
    Function for triggering training of the model
    """
    print("request received: training the model")
    results = train("Hello! This is the result of training from Flask backend!")
    resp = jsonify(results)
    resp.headers.add("Access-Control-Allow-Origin", "*")
    return resp

@app.route("/analyze", methods = ["POST"])
def analyze_it():
    """
    Function for triggering training of the model
    """
    print("request received: analzing the model")
    resp = jsonify(analyze("Hello! This is the result of analysis from Flask backend!"))
    resp.headers.add("Access-Control-Allow-Origin", "*")
    return resp

if __name__ == "__main__":
	   app.run(HOST, PORT)