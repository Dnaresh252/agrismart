from django.shortcuts import render
from django.http import JsonResponse

# Health check endpoint for Render

def health_check(request):
    return JsonResponse({"status": "ok"})

def CropRecommend(request):
    if request.method == 'GET':
        return render(request, 'CropRecommend.html', {})

def index(request):
    if request.method == 'GET':
        return render(request, 'index.html', {})

def getCNNModel():
    import pandas as pd
    import numpy as np
    import pickle
    import os
    from sklearn.preprocessing import LabelEncoder, normalize
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    from tensorflow.keras.models import Sequential, model_from_json
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

    le = LabelEncoder()
    dataset = pd.read_csv('dataset/Crop_recommendation.csv', usecols=['N','P','K','ph','rainfall','label'])
    dataset.fillna(0, inplace=True)
    labels = dataset['label']
    dataset['label'] = pd.Series(le.fit_transform(labels.astype(str)))
    Y = dataset.values[:, 5]
    dataset.drop(['label'], axis=1, inplace=True)
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
        model.load_weights("model/cnnmodel_weights.h5")
    else:
        model = Sequential()
        model.add(Conv2D(64, (1, 1), input_shape=(XX.shape[1], XX.shape[2], XX.shape[3]), activation='relu'))
        model.add(MaxPooling2D(pool_size=(1, 1)))
        model.add(Conv2D(32, (1, 1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(1, 1)))
        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(Y.shape[1], activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        hist = model.fit(XX, Y, batch_size=16, epochs=100, shuffle=True, verbose=2, validation_data=(X_test, y_test))
        with open('model/cnnhistory.pckl', 'wb') as f:
            pickle.dump(hist.history, f)
        model.save_weights('model/cnnmodel_weights.h5')
        model_json = model.to_json()
        with open("model/cnnmodel.json", "w") as json_file:
            json_file.write(model_json)

    predict = model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_test, predict)
    acc = accuracy_score(y_test, predict) * 100
    p = precision_score(y_test, predict, average='macro') * 100
    r = recall_score(y_test, predict, average='macro') * 100
    f = f1_score(y_test, predict, average='macro') * 100
    return cm, acc, p, r, f

def getLSTMModel():
    import pandas as pd
    import numpy as np
    import pickle
    import os
    from sklearn.preprocessing import LabelEncoder, normalize
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    from tensorflow.keras.models import Sequential, model_from_json
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional

    le = LabelEncoder()
    dataset = pd.read_csv('dataset/Crop_recommendation.csv', usecols=['N','P','K','ph','rainfall','label'])
    dataset.fillna(0, inplace=True)
    labels = dataset['label']
    dataset['label'] = pd.Series(le.fit_transform(labels.astype(str)))
    Y = dataset.values[:, 5]
    dataset.drop(['label'], axis=1, inplace=True)
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
        model = model_from_json(loaded_model_json)
        model.load_weights("model/lstmmodel_weights.h5")
        with open('model/lstmhistory.pckl', 'rb') as f:
            data = pickle.load(f)
        accuracy = data[9] * 100
    else:
        model = Sequential()
        model.add(Bidirectional(LSTM(32, input_shape=(XX.shape[1],1), activation='relu', return_sequences=True)))
        model.add(Dropout(0.2))
        model.add(Bidirectional(LSTM(32, activation='relu')))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(Y.shape[1], activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        hist = model.fit(XX, Y, epochs=100, batch_size=64)
        with open('model/lstmhistory.pckl', 'wb') as f:
            pickle.dump(hist.history['accuracy'], f)
        model.save_weights('model/lstmmodel_weights.h5')
        model_json = model.to_json()
        with open("model/lstmmodel.json", "w") as json_file:
            json_file.write(model_json)
        accuracy = hist.history['accuracy'][9] * 100

    predict = model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_test, predict)
    acc = accuracy_score(y_test, predict) * 100
    p = precision_score(y_test, predict, average='macro') * 100
    r = recall_score(y_test, predict, average='macro') * 100
    f = f1_score(y_test, predict, average='macro') * 100
    return cm, acc, p, r, f

def LoadModel(request):
    if request.method == 'GET':
        cnn_cm, cnn_accuracy, cnn_precision, cnn_recall, cnn_fscore = getCNNModel()
        lstm_cm, lstm_accuracy, lstm_precision, lstm_recall, lstm_fscore = getLSTMModel()
        output = '<table border=1 align=center>'
        output += '<tr><th>Algorithm</th><th>Confusion Matrix</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1 Score</th></tr>'
        output += f'<tr><td>CNN</td><td>{cnn_cm}</td><td>{cnn_accuracy:.2f}</td><td>{cnn_precision:.2f}</td><td>{cnn_recall:.2f}</td><td>{cnn_fscore:.2f}</td></tr>'
        output += f'<tr><td>LSTM</td><td>{lstm_cm}</td><td>{lstm_accuracy:.2f}</td><td>{lstm_precision:.2f}</td><td>{lstm_recall:.2f}</td><td>{lstm_fscore:.2f}</td></tr>'
        output += '</table><br/><br/><br/><br/><br/>'
        return render(request, 'TrainDL.html', {'data': output})
