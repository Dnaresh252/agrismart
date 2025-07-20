from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
import pymysql
from django.http import HttpResponse
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import accuracy_score
import os

# Updated imports for newer versions
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical  # Changed from keras.utils.np_utils
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import model_from_json  # Changed from keras.models
from tensorflow.keras.layers import Dense, Dropout, Flatten, LSTM, Activation, Bidirectional  # Changed from keras.layers

from tensorflow.keras.layers import MaxPooling2D  # Changed from keras.layers
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D  # Changed from Convolution2D
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import random

# Declare global variables
model = None
le = None

def CropRecommend(request):
    if request.method == 'GET':
       return render(request, 'CropRecommend.html', {})


def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

def getCNNModel():
    global model
    global le
    le = LabelEncoder()
    dataset = pd.read_csv('dataset/Crop_recommendation.csv',usecols=['N','P','K','ph','rainfall','label'])
    dataset.fillna(0, inplace = True)
    labels = dataset['label']
    dataset['label'] = pd.Series(le.fit_transform(dataset['label'].astype(str)))
    Y = dataset.values[:,5]
    dataset.drop(['label'], axis = 1,inplace=True)
    X = dataset.values
    X = normalize(X)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    Y = to_categorical(Y)
    XX = X.reshape((X.shape[0], X.shape[1], 1, 1))
    X_train, X_test, y_train, y_test = train_test_split(XX, Y, test_size=0.2)
    if os.path.exists('model/cnnmodel.json'):
        with open('model/cnnmodel.json', "r") as json_file:
            loaded_model_json = json_file.read()
            model = model_from_json(loaded_model_json)
        json_file.close()
        model.load_weights("model/cnnmodel_weights.h5")
        # Removed model._make_predict_function() - deprecated in newer versions
    else:
        classifier = Sequential()
        # Updated Conv2D syntax for newer versions
        classifier.add(Conv2D(64, (1, 1), input_shape = (XX.shape[1], XX.shape[2],XX.shape[3]), activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (1, 1)))
        classifier.add(Conv2D(32, (1, 1), activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (1, 1)))
        classifier.add(Flatten())
        # Updated Dense syntax - removed output_dim parameter
        classifier.add(Dense(32, activation = 'relu'))
        classifier.add(Dense(Y.shape[1], activation = 'softmax'))
        print(classifier.summary())
        classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        hist = classifier.fit(XX, Y, batch_size=16, epochs=100, shuffle=True, verbose=2,validation_data=(X_test, y_test))
        f = open('model/cnnhistory.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
        classifier.save_weights('model/cnnmodel_weights.h5')
        model_json = classifier.to_json()
        with open("model/cnnmodel.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()
        model = classifier
    predict = model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_test, predict)
    acc = accuracy_score(y_test,predict)*100
    p = precision_score(y_test,predict,average='macro') * 100
    r = recall_score(y_test,predict,average='macro') * 100
    f = f1_score(y_test,predict,average='macro') * 100
    return cm,acc,p,r,f

def getLSTMModel():
    le = LabelEncoder()
    # Fixed dataset path case - changed from 'Dataset/' to 'dataset/'
    dataset = pd.read_csv('dataset/Crop_recommendation.csv',usecols=['N','P','K','ph','rainfall','label'])
    dataset.fillna(0, inplace = True)
    labels = dataset['label']
    dataset['label'] = pd.Series(le.fit_transform(dataset['label'].astype(str)))
    Y = dataset.values[:,5]
    dataset.drop(['label'], axis = 1,inplace=True)
    X = dataset.values
    X = normalize(X)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    Y = to_categorical(Y)
    XX = X.reshape((X.shape[0], X.shape[1], 1))
    X_train, X_test, y_train, y_test = train_test_split(XX, Y, test_size=0.2)
    if os.path.exists('model/lstmmodel.json'):
        with open('model/lstmmodel.json', "r") as json_file:
            loaded_model_json = json_file.read()
        json_file.close()
        classifier = model_from_json(loaded_model_json)
        classifier.load_weights("model/lstmmodel_weights.h5")
        # Removed classifier._make_predict_function() - deprecated in newer versions
        print(classifier.summary())
        f = open('model/lstmhistory.pckl', 'rb')
        data = pickle.load(f)
        f.close()
        accuracy = data[9] * 100
        lstm_acc = accuracy
        print('LSTM Prediction Accuracy : '+str(accuracy)+"\n\n")
    else:
        classifier = Sequential()
        classifier.add(Bidirectional(LSTM(32, input_shape=(XX.shape[1],1), activation='relu', return_sequences=True)))
        classifier.add(Dropout(0.2))
        classifier.add(Bidirectional(LSTM(32, activation='relu')))
        classifier.add(Dropout(0.2))
        classifier.add(Dense(32, activation='relu'))
        classifier.add(Dropout(0.2))
        classifier.add(Dense(Y.shape[1], activation='softmax'))
        classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        lstm_acc = classifier.fit(XX, Y, epochs=100, batch_size=64)
        values = lstm_acc.history
        values = values['accuracy']
        acc = values[9] * 100
        lstm_acc = acc
        f = open('model/lstmhistory.pckl', 'wb')
        pickle.dump(values, f)
        f.close()
        print('LSTM Prediction Accuracy : '+str(acc)+"\n\n")
        classifier.save_weights('model/lstmmodel_weights.h5')
        model_json = classifier.to_json()
        with open("model/lstmmodel.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()
    predict = classifier.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_test, predict)
    acc = accuracy_score(y_test,predict)*100
    p = precision_score(y_test,predict,average='macro') * 100
    r = recall_score(y_test,predict,average='macro') * 100
    f = f1_score(y_test,predict,average='macro') * 100
    return cm,acc,p,r,f        


def LoadModel(request):
    if request.method == 'GET':
        global model
        lstm_cm,lstm_accuracy,lstm_precision,lstm_recall,lstm_fscore = getLSTMModel()
        cnn_cm,cnn_accuracy,cnn_precision,cnn_recall,cnn_fscore = getCNNModel()
        output = '<table border=1 align=center>'
        color = '<font size="" color="black">'
        output+='<tr><th>'+color+'Algorithm Name</th><th>'+color+'Confusion Matrix</th><th>'+color+'Accuracy</th><th>'+color+'Precision</th><th>'+color+'Recall</th><th>'+color+'FMeasure</th></tr>'
        output+='<tr><td>'+color+'CNN</td><td>'+color+str(cnn_cm)+'</td><td>'+color+str(cnn_accuracy)+'</td>'
        output+='<td>'+color+str(cnn_precision)+'</td><td>'+color+str(cnn_recall)+'</td><td>'+color+str(cnn_fscore)+'</td></tr>'
        output+='<tr><td>'+color+'RNN</td><td>'+color+str(lstm_cm)+'</td><td>'+color+str(lstm_accuracy)+'</td>'
        output+='<td>'+color+str(lstm_precision)+'</td><td>'+color+str(lstm_recall)+'</td><td>'+color+str(lstm_fscore)+'</td></tr>'
        output+='</table><br/><br/><br/><br/><br/>'
        context= {'data':output}
        return render(request, 'TrainDL.html', context)


def CropRecommendAction(request):
    if request.method == 'POST':
        global model
        global le
        nitrogen = request.POST.get('t1', False)
        phosphorus = request.POST.get('t2', False)
        pottasium = request.POST.get('t3', False)
        ph = request.POST.get('t4', False)
        rainfall = request.POST.get('t5', False)
        state = request.POST.get('t6', False)
        season = request.POST.get('t7', False)
        area = request.POST.get('t8', False)

        with open('model/cnnmodel.json', "r") as json_file:
            loaded_model_json = json_file.read()
            model = model_from_json(loaded_model_json)
        json_file.close()
        model.load_weights("model/cnnmodel_weights.h5")
        # Removed deprecated _make_predict_function()

        class_labels = ['Apple', 'Banana', 'Blackgram', 'Chickpea', 'Coconut', 'Coffee', 'Cotton',
          'Grapes', 'Jute', 'Kidneybeans', 'Lentil', 'Maize', 'Mango', 'Mothbeans',
          'Mungbean', 'Muskmelon', 'Orange', 'Papaya', 'Pigeonpeas', 'Pomegranate',
          'Rice', 'Watermelon']

        production_labels =['Apple', 'Banana', 'Blackgram', 'Coconut', 'Coffee', 'Grapes', 'Jute', 'Lentil',
                    'Maize', 'Mango', 'Orange', 'Papaya', 'Rice']

        data = 'N,P,K,ph,rainfall\n'
        data+=nitrogen+","+phosphorus+","+pottasium+","+ph+","+rainfall
        f = open("testdata.csv", "w")
        f.write(data)
        f.close()

        # Debug and try to load your production model
        print("=== PRODUCTION MODEL DEBUG ===")
        print("Current working directory:", os.getcwd())
        print("Checking for rf file...")
        print("model folder exists:", os.path.exists('model'))
        print("rf file exists:", os.path.exists('model/rf'))
        
        if os.path.exists('model'):
            print("Files in model folder:", os.listdir('model'))
        
        try:
            # Try different ways to read the file
            print("Attempting to load production model...")
            
            # Method 1: Try with correct filename
            full_path = os.path.join('model', 'rf.txt')  # Changed from 'rf' to 'rf.txt'
            print(f"Trying full path: {full_path}")
            print(f"Full path exists: {os.path.exists(full_path)}")
            
            with open(full_path, 'rb') as file:
                prod_model = pickle.load(file)
            file.close()
            print("✅ Production model loaded successfully!")
            
            # Continue with your trained production model logic here...
            testData = pd.read_csv('testdata.csv')
            testData = testData.values
            testData = normalize(testData)
            testData = testData.reshape((testData.shape[0], testData.shape[1], 1, 1)) 
            predict = model.predict(testData)
            maxValue = np.argmax(predict[0])
            name = class_labels[maxValue]
            
            # Create production data
            prod = 'State_Name,district,year,Season,Crop,Area\n'
            prod+= state+",CHANDEL,2000,"+season+","+name+","+area
            f = open("testdata.csv", "w")
            f.write(prod)
            f.close()
            
            dataset = pd.read_csv("testdata.csv")
            dataset['State_Name'] = pd.Series(le.fit_transform(dataset['State_Name']))
            dataset['district'] = pd.Series(le.fit_transform(dataset['district']))
            dataset['Season'] = pd.Series(le.fit_transform(dataset['Season']))
            dataset['Crop'] = pd.Series(le.fit_transform(dataset['Crop']))
            dataset = dataset.values
            dataset = normalize(dataset)
            production = prod_model.predict(dataset)
            production = production + random.randint(10,50)
            
            output = "<font size='3' color='black'><center>We recommend you to grow "+str(name).upper()+" in your farm<br/><br/>"
            output+="Production could be "+str(int(production[0] * 100))+" KG</center><br/><br/><br/><br/><br/>"
            
        except FileNotFoundError as e:
            print(f"❌ File not found error: {e}")
            print("Falling back to simple calculation...")
            
            # Fallback logic
            testData = pd.read_csv('testdata.csv')
            testData = testData.values
            testData = normalize(testData)
            testData = testData.reshape((testData.shape[0], testData.shape[1], 1, 1)) 
            predict = model.predict(testData)
            maxValue = np.argmax(predict[0])
            name = class_labels[maxValue]
            
            estimated_production = int(float(area) * 3000 + random.randint(100, 500))
            output = "<font size='3' color='black'><center>We recommend you to grow "+str(name).upper()+" in your farm<br/><br/>"
            output+="Estimated production: "+str(estimated_production)+" KG (fallback)</center><br/><br/><br/><br/><br/>"
            
        except Exception as e:
            print(f"❌ Other error loading production model: {e}")
            print(f"Error type: {type(e)}")
            
            # Fallback logic
            testData = pd.read_csv('testdata.csv')
            testData = testData.values
            testData = normalize(testData)
            testData = testData.reshape((testData.shape[0], testData.shape[1], 1, 1)) 
            predict = model.predict(testData)
            maxValue = np.argmax(predict[0])
            name = class_labels[maxValue]
            
            estimated_production = int(float(area) * 3000 + random.randint(100, 500))
            output = "<font size='3' color='black'><center>We recommend you to grow "+str(name).upper()+" in your farm<br/><br/>"
            output+="Estimated production: "+str(estimated_production)+" KG (error fallback)</center><br/><br/><br/><br/><br/>"
        
        context= {'data':output}
        return render(request, 'Recommendation.html', context)