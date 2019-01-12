import pandas as pd
import os
import datetime
import zipfile

begin = datetime.datetime.now()

## Caminho original do arquivo .py
file_path = os.path.dirname(os.path.abspath(__file__))

dataset_folder = 'dataset'
zip_file = 'dataset.zip'

train_file = 'train.csv'
test_file = 'test.csv'

if os.path.isfile(os.path.join(file_path,dataset_folder,train_file)):
    with zipfile.ZipFile(os.path.join(file_path, dataset_folder,zip_file), 'r') as zip:
        zip.extractall(os.path.join(file_path, dataset_folder))

## Caminho dos dataset's de treino e teste
train_path = os.path.join(file_path, dataset_folder, train_file)
test_path = os.path.join(file_path, dataset_folder, test_file)

print(datetime.datetime.now(),' - Load csv files ')
df_train_original = pd.read_csv(train_path, sep=';', header=None, names=['Body','Tag','Title'])
df_test_original = pd.read_csv(test_path, sep=';', header=None, names=['Body','Tag','Title'])

## Tratamento de dados 
from nltk.corpus import stopwords 
from nltk import word_tokenize
import re

stopWords_original = stopwords.words('english')
stopWords = []

## Retira caracteres especiais da lista de stopwords
for word in stopWords_original:
    stopWords.append(re.sub('[^a-z]+','',word.lower()))

df_test = pd.DataFrame()
df_train = pd.DataFrame()

# ### 
# # Retira caracteres especiais, tags de html e stopwords do dataset  
# # Também faz a união do Title com o Body para uma melhor accuracy na hora do treinamento
# ###

print(datetime.datetime.now(),' - Tratamento de dados: testes')
## Dataset de Teste (Tratamento)
for index, rows in df_test_original.iterrows():

    title = word_tokenize(re.sub('[^a-z]+',' ',rows['Title'].lower()))
    body = word_tokenize(re.sub('[^a-z]+',' ',re.sub('<[^>]+>',' ',rows['Body'].lower())))

    content = []
    content.append([w for w in title + body if w not in stopWords])

    data = []
    data.append((' '.join(content[0]), rows['Tag']))

    df_test = pd.concat([df_test, pd.DataFrame(data)], ignore_index=True)


print(datetime.datetime.now(),' - Tratamento de dados: treino')
## Dataset de Treino (Tratamento)
for index, rows in df_train_original.iterrows():

    title = word_tokenize(re.sub('[^a-z]+',' ',rows['Title'].lower()))
    body = word_tokenize(re.sub('[^a-z]+',' ',re.sub('<[^>]+>',' ',rows['Body'].lower())))

    content = []
    content.append([w for w in title + body if w not in stopWords])

    data = []
    data.append((' '.join(content[0]), rows['Tag']))

    df_train = pd.concat([df_train, pd.DataFrame(data)], ignore_index=True)

df_test.columns = ['Content','Tag']
df_train.columns = ['Content', 'Tag']

## Pré-processamento

from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from keras import utils
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
import numpy as np

# # #
# # Numero de palavras maxima para o modelo de treinamento (top N palavras)
# # Caso o valor seja aumentado, irá aumentar também o consumo de memoria da sua maquina
# # Fique ligado a quanto de memoria você poderá utilizar no treinamento, caso contrato irá ocorrer erro de memoria
# # #
num_Words = 1000

tokenizer = Tokenizer(num_words=num_Words)
tokenizer.fit_on_texts(df_train['Content'])

print(datetime.datetime.now(),' - Texts to matrix')
x_train = tokenizer.texts_to_matrix(df_train['Content'])
x_test = tokenizer.texts_to_matrix(df_test['Content'])

encoder = LabelEncoder()
encoder.fit(df_train['Tag'])

print(datetime.datetime.now(),' - Encoder transform')
y_train = encoder.transform(df_train['Tag'])
y_test = encoder.transform(df_test['Tag'])

## Numero de classes para serem classificadas (Nesse caso, o numero de Tags unicas)
n_classes = np.max(y_test) + 1

print(datetime.datetime.now(),' - Categorical')
y_train = utils.to_categorical(y_train, num_classes=n_classes)
y_test = utils.to_categorical(y_test, num_classes=n_classes)

## Numero de exemplos a serem treinados por vez
batch_size = 100
## Numero de vezes que a rede neural irá treinar todo o dataset
epochs = 10

model = Sequential()
## Layer 1 com 521 neuronios com o input de shape (word_max, )
model.add(Dense(512, input_shape=(num_Words,)))
model.add(Activation('relu'))
## Dropout para evitar overfitting
model.add(Dropout(0.5))
## Output layer com o numero de classes do modelo
model.add(Dense(n_classes))
model.add(Activation('softmax'))

print(datetime.datetime.now(),' - Compile')
model.compile(  loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

print(datetime.datetime.now(),' - Fit')
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2,
                    validation_split=0.1)


labels = encoder.classes_

print(datetime.datetime.now(),' - Evaluate')
score = model.evaluate(x_test, y_test,
                    batch_size= batch_size,
                    verbose=1)

print('Test score: ', score[0])
print('Test accuracy: ',score[1])

## Teste com 10 exemplos do dataset
for n in range(0,10):
    prediction = model.predict(np.array([x_test[n]]))
    predicted_label = labels[np.argmax(prediction)]
    print('Question: '+ df_test_original['Title'].iloc[n])
    print('Actual label:' + df_test_original['Tag'].iloc[n])
    print("Predicted label: " + predicted_label + "\n")
    print('\n')

end = datetime.datetime.now()

diff = end - begin

print('Tempo estimado de execução: %d horas e %d minutos'%(diff.seconds/3600, diff.seconds/60)

###
# Você pode salvar o modelo gerado em um arquivo para utilizar em uma classificação sem precisar re-treinar
# É importante tambem salvar seus labels e o tokenizer
# O label é salvo em um arquivo .npy (numpy), o tokenizer em um arquivo .pickle (pickle) e o modelo em um arquivo .h5 (HDF5)
# ** Caso grave o arquivo labels ou tokenizer em outro formato, tenha certeza que os arquivos estejam com suas propriedades na forma original 
# ** em que foram treinadas
# ** Qualquer ordem que não estejam conforme os treinamentos iram retornar resultados errados na predição
# ###

import pickle

model.save('model.h5')
np.save('labels.npy',labels)
with open('tokenizer.pickle','wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Para re-utilizar os arquivos do treinamento, apenas carregue eles para o modelo

from keras.models import load_model

model = load_model('model.h5')
labels = np.load('labels.npy')
with open('tokenizer.pickle','rb') as handle:
    tokenizer = pickle.load(handle)

texts = ['How can I redirect the user from one page to another using jQuery or pure JavaScript?',
        'What is the difference between “INNER JOIN” and “OUTER JOIN” in SQL Server?',
        'How do you give a C# Auto-Property a default value? I either use the constructor, or revert to the old syntax.',
        'In PHP 5, what is the difference between using self and $this? When is each appropriate?',
        'What are valid values for the id attribute in HTML?',
        'Optimizing Lucene performance',
        'Continuous Integration System for Delphi',
        'How does database indexing work?']

for t in texts:

    data = []
    data.append(t)
    X = tokenizer.texts_to_matrix(data)
    prediction = model.predict(X)
    predicted_label = labels[np.argmax(prediction)]
    print('Question: ',t)
    print('Prediction Tag: ',predicted_label)
    print('\n')