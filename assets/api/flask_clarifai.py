from flask import Flask
from flask import jsonify
import base64
import json
import urllib as urllib2
from clarifai.rest import ClarifaiApp

HOST = '127.0.0.1'
PORT = '5000'

app = Flask(__name__)

print('********************************')
print('Setting up Clarifai API\n')

clarifai_app = ClarifaiApp(api_key='cd87337f00f64ab6b2d75734d26636b5')
model = 'general-v1.3'
model = clarifai_app.models.get(model)


print('********************************')
print('Setting up LDA Modeller\n')
import lda_model as lda

@app.route("/image_ext/<string:image>")
def getImageTagsExternal(image):

    d_res = ''

    try:
        print('Received external image request')
        print('URL: ' + str(base64.b64decode(image)))
        res = model.predict_by_url(url=(base64.b64decode(image).decode("utf-8")))
        print(res)
        d_res = json.dumps(res)

        print(d_res)

    except Exception as e:
        print('Error encountered.')
        print(e)

    response = jsonify({'data': d_res})
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response


@app.route("/image/<string:image>")
def getImageTags(image):
    print('Received image request')

    tags = getTags(image)

    response = jsonify({'data': tags})
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

@app.route("/lda/<string:text>")
def getTopics(text):
    print('Received LDA request')


    text = base64.b64decode(text).decode("unicode-escape")
    text = (json.loads(text))

    try: text = [urllib2.request.unquote(urllib2.request.unquote(s)) for s in text]
    except: text = [urllib2.unquote(urllib2.unquote(s)) for s in text]

    print('Number of documents: ' + str(len(text)))

    output = lda.get_topics(text)

    response = jsonify({'data': output})

    print(output)


    response.headers.add('Access-Control-Allow-Origin', '*')

    return response



def getTags(image):

    image = (base64.b64decode(image))

    res = model.predict_by_base64(image)

    d_res = json.dumps(res)

    print(d_res)

    return d_res


if __name__ == "__main__":
	   app.run(HOST,PORT)
