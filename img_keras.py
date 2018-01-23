from __future__ import print_function
from flask import Flask, request
import json
import requests
#import urllib3.request as request
import os
from cloudant import Cloudant
import cf_deployment_tracker
import atexit
import uuid # to generate random file names!
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

app = Flask(__name__)
with open('bottoken.txt','r') as fh:
    bottoken = fh.read()
baseURL = 'https://api.telegram.org/bot'
fileBaseURL = 'https://api.telegram.org/file/bot'

print ('loading pre-trained ResNet50 model...')
model = ResNet50(weights='imagenet')
print ('Ready to make predictions.\n')

# --- IBM cloud config stuff starts

cf_deployment_tracker.track()
db_name = 'mydb'
client = None
db = None

if 'VCAP_SERVICES' in os.environ:
    vcap = json.loads(os.getenv('VCAP_SERVICES'))
    print('Found VCAP_SERVICES')
    if 'cloudantNoSQLDB' in vcap:
        creds = vcap['cloudantNoSQLDB'][0]['credentials']
        user = creds['username']
        password = creds['password']
        url = 'https://' + creds['host']
        client = Cloudant(user, password, url=url, connect=True)
        db = client.create_database(db_name, throw_on_exists=False)
elif os.path.isfile('vcap-local.json'):
    with open('vcap-local.json') as f:
        vcap = json.load(f)
        print('Found local VCAP_SERVICES')
        creds = vcap['services']['cloudantNoSQLDB'][0]['credentials']
        user = creds['username']
        password = creds['password']
        url = 'https://' + creds['host']
        client = Cloudant(user, password, url=url, connect=True)
        db = client.create_database(db_name, throw_on_exists=False)

@atexit.register
def shutdown():
    if client:
        client.disconnect()
# --- IBM stuff ends

def composeMsg(recvmsg):
    if recvmsg[0] == '/':
        return "..."
    # get result .. with what??
    
    return "Under construction"
    
    
# ----------------------------------------------    
# common stuff 
# ----------------------------------------------
    
@app.route('/',methods=['POST'])
def listener():
    data = json.loads(request.data.decode())
    message = data["message"]
    keys = message.keys()
    chat_id = message['chat']['id']
    
    if 'text' in keys:
        user_said = message['text']
        if user_said.startswith('/'):
            sendReply(chat_id, "A command? whatever")
        else:
            sendReply(chat_id, "Too busy deploying ResNet50. Try sending an image instead!")
        
    elif 'photo' in keys:
        file_id = message['photo'][-1]['file_id']
        fdata = { 'file_id' : file_id }
        f_url = baseURL + bottoken + '/getfile'
        f_obj = requests.get(f_url, params = fdata)
        response = json.loads(f_obj.text)
        if response['ok'] == True:
            resultDict = response['result']
            file_path = resultDict['file_path']
            f_url = fileBaseURL + bottoken + '/' + file_path
            savefilename = 'recvImg_' + uuid.uuid4().hex[:13] + '.jpg'
            #request.urlretrieve(f_url, savefilename)
            img_data = requests.get(f_url)
            with open(savefilename, 'wb') as f:
                f.write(img_data.content)
            print ("Succesfully saved image to : ", savefilename)
            # now the actual thing...which will take place elsewhere
            predicted_class, probability = classifyImage(savefilename)
            response = "Hello " + message['chat']['first_name'] + ", I think that is a " + predicted_class + " with probability {0:.3f}".format(probability)
            newname = predicted_class + '_{0:.3f}'.format(probability)+"_from_"+message['chat']['first_name']+'.jpg'
            os.rename(savefilename, newname)
            print (response)
            sendReply(chat_id, response)
        
        else:
            # "Something went wrong
            sendReply(chat_id, 'Error retrieving message')
    
    
    else:
        sendReply(chat_id, "I don't know what you sent me")
    
    return request.data
    

def sendReply(chat_id, sendmsg):
    data = { "chat_id" : chat_id, "text" : sendmsg }
    payload = json.dumps(data)
    sendURL = baseURL + bottoken + "/sendMessage"
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    resp = requests.post(sendURL, data=payload, headers=headers)
 
def classifyImage(fname):
    img = image.load_img(fname, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    pred = decode_predictions(model.predict(x), top=1)[0][0]
    
    return (pred[1], pred[2])
 
port = int(os.getenv('PORT', 8000))
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)
   
    
    
    