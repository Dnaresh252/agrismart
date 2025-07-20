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
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Dense, Dropout, Flatten, LSTM, Activation, Bidirectional
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D
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
        print("Loading existing CNN model...")
        with open('model/cnnmodel.json', "r") as json_file:
            loaded_model_json = json_file.read()
            model = model_from_json(loaded_model_json)
        json_file.close()
        model.load_weights("model/cnnmodel_weights.h5")
        print("✅ CNN model loaded successfully!")
    else:
        print("Training new CNN model...")
        classifier = Sequential()
        classifier.add(Conv2D(64, (1, 1), input_shape = (XX.shape[1], XX.shape[2],XX.shape[3]), activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (1, 1)))
        classifier.add(Conv2D(32, (1, 1), activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (1, 1)))
        classifier.add(Flatten())
        classifier.add(Dense(32, activation = 'relu'))
        classifier.add(Dense(Y.shape[1], activation = 'softmax'))
        print(classifier.summary())
        classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        hist = classifier.fit(XX, Y, batch_size=16, epochs=50, shuffle=True, verbose=2,validation_data=(X_test, y_test))
        
        # Save the model
        try:
            os.makedirs('model', exist_ok=True)
            model_json = classifier.to_json()
            with open("model/cnnmodel.json", "w") as json_file:
                json_file.write(model_json)
            json_file.close()
            classifier.save_weights('model/cnnmodel_weights.h5')
            f = open('model/cnnhistory.pckl', 'wb')
            pickle.dump(hist.history, f)
            f.close()
            print("✅ CNN model saved successfully!")
        except Exception as e:
            print(f"⚠️ Warning: Could not save CNN model: {e}")
        
        model = classifier
        
    predict = model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_test, predict)
    acc = accuracy_score(y_test,predict)*100
    p = precision_score(y_test,predict,average='macro') * 100
    r = recall_score(y_test,predict,average='macro') * 100
    f = f1_score(y_test,predict,average='macro') * 100
    print(f"CNN Model Accuracy: {acc:.2f}%")
    return cm,acc,p,r,f

def getLSTMModel():
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
    XX = X.reshape((X.shape[0], X.shape[1], 1))
    X_train, X_test, y_train, y_test = train_test_split(XX, Y, test_size=0.2)
    
    if os.path.exists('model/lstmmodel.json'):
        print("Loading existing LSTM model...")
        with open('model/lstmmodel.json', "r") as json_file:
            loaded_model_json = json_file.read()
        json_file.close()
        classifier = model_from_json(loaded_model_json)
        classifier.load_weights("model/lstmmodel_weights.h5")
        print("✅ LSTM model loaded successfully!")
    else:
        print("Training new LSTM model...")
        classifier = Sequential()
        classifier.add(Bidirectional(LSTM(32, input_shape=(XX.shape[1],1), activation='relu', return_sequences=True)))
        classifier.add(Dropout(0.2))
        classifier.add(Bidirectional(LSTM(32, activation='relu')))
        classifier.add(Dropout(0.2))
        classifier.add(Dense(32, activation='relu'))
        classifier.add(Dropout(0.2))
        classifier.add(Dense(Y.shape[1], activation='softmax'))
        classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        lstm_acc = classifier.fit(XX, Y, epochs=50, batch_size=64, verbose=1)
        
        # Save the model
        try:
            os.makedirs('model', exist_ok=True)
            model_json = classifier.to_json()
            with open("model/lstmmodel.json", "w") as json_file:
                json_file.write(model_json)
            json_file.close()
            classifier.save_weights('model/lstmmodel_weights.h5')
            values = lstm_acc.history['accuracy']
            f = open('model/lstmhistory.pckl', 'wb')
            pickle.dump(values, f)
            f.close()
            print("✅ LSTM model saved successfully!")
        except Exception as e:
            print(f"⚠️ Warning: Could not save LSTM model: {e}")
    
    predict = classifier.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_test, predict)
    acc = accuracy_score(y_test,predict)*100
    p = precision_score(y_test,predict,average='macro') * 100
    r = recall_score(y_test,predict,average='macro') * 100
    f = f1_score(y_test,predict,average='macro') * 100
    print(f"LSTM Model Accuracy: {acc:.2f}%")
    return cm,acc,p,r,f

def LoadModel(request):
    if request.method == 'GET':
        global model
        print("Starting model loading process...")
        lstm_cm,lstm_accuracy,lstm_precision,lstm_recall,lstm_fscore = getLSTMModel()
        cnn_cm,cnn_accuracy,cnn_precision,cnn_recall,cnn_fscore = getCNNModel()
        print("All models loaded successfully!")
        
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
        
        print("=== CROP RECOMMENDATION ===")
        
        nitrogen = request.POST.get('t1', False)
        phosphorus = request.POST.get('t2', False)
        pottasium = request.POST.get('t3', False)
        ph = request.POST.get('t4', False)
        rainfall = request.POST.get('t5', False)
        state = request.POST.get('t6', False)
        season = request.POST.get('t7', False)
        area = request.POST.get('t8', False)
        
        print(f"Input values: N={nitrogen}, P={phosphorus}, K={pottasium}, pH={ph}, Rainfall={rainfall}")
        print(f"State={state}, Season={season}, Area={area}")

        # Load the CNN model for prediction
        try:
            print("Loading CNN model for prediction...")
            with open('model/cnnmodel.json', "r") as json_file:
                loaded_model_json = json_file.read()
                model = model_from_json(loaded_model_json)
            json_file.close()
            model.load_weights("model/cnnmodel_weights.h5")
            print("✅ CNN model loaded for prediction!")
        except Exception as e:
            print(f"❌ Error loading model for prediction: {e}")
            output = "<font size='3' color='red'><center>Error: Model not available. Please train the model first.</center><br/><br/><br/><br/><br/>"
            context= {'data':output}
            return render(request, 'Recommendation.html', context)

        class_labels = ['Apple', 'Banana', 'Blackgram', 'Chickpea', 'Coconut', 'Coffee', 'Cotton',
          'Grapes', 'Jute', 'Kidneybeans', 'Lentil', 'Maize', 'Mango', 'Mothbeans',
          'Mungbean', 'Muskmelon', 'Orange', 'Papaya', 'Pigeonpeas', 'Pomegranate',
          'Rice', 'Watermelon']

        # Create test data
        data = 'N,P,K,ph,rainfall\n'
        data+=nitrogen+","+phosphorus+","+pottasium+","+ph+","+rainfall
        f = open("testdata.csv", "w")
        f.write(data)
        f.close()

        # Make crop prediction
        testData = pd.read_csv('testdata.csv')
        testData = testData.values
        testData = normalize(testData)
        testData = testData.reshape((testData.shape[0], testData.shape[1], 1, 1)) 
        predict = model.predict(testData)
        maxValue = np.argmax(predict[0])
        name = class_labels[maxValue]
        print(f"Predicted crop: {name}")
        
        # Smart production calculation based on crop type and area
        production_multiplier = {
            'APPLE': 8000, 'BANANA': 6000, 'BLACKGRAM': 1200, 'CHICKPEA': 1800, 
            'COCONUT': 2000, 'COFFEE': 1200, 'COTTON': 1500, 'GRAPES': 5000, 
            'JUTE': 2500, 'KIDNEYBEANS': 2200, 'LENTIL': 1500, 'MAIZE': 4000, 
            'MANGO': 7000, 'MOTHBEANS': 1300, 'MUNGBEAN': 1400, 'MUSKMELON': 3500, 
            'ORANGE': 6500, 'PAPAYA': 4500, 'PIGEONPEAS': 1600, 'POMEGRANATE': 5500, 
            'RICE': 3000, 'WATERMELON': 4000
        }
        
        try:
            crop_multiplier = production_multiplier.get(name.upper(), 3000)
            estimated_production = int(float(area) * crop_multiplier + random.randint(100, 500))
            print(f"Estimated production: {estimated_production} KG")
            
            output = "<font size='3' color='black'><center>We recommend you to grow "+str(name).upper()+" in your farm<br/><br/>"
            output+="Production could be "+str(estimated_production)+" KG</center><br/><br/><br/><br/><br/>"
        except Exception as e:
            print(f"Error in production calculation: {e}")
            output = "<font size='3' color='black'><center>We recommend you to grow "+str(name).upper()+" in your farm<br/><br/>"
            output+="Production calculation error</center><br/><br/><br/><br/><br/>"
        
        print("Recommendation completed!")
        context= {'data':output}
        return render(request, 'Recommendation.html', context)