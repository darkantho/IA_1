import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt


#1142


f=open("/path/granos/granos_train_test.csv","r")

granos=[]
valores=[]

for i in f:
  linea=i.strip().split(";")
  if linea[-1]!="Label":
     granos.append(list(map(float,linea[0:len(linea)-1])))
     valores.append(int(linea[-1]))
     

granos=np.array(granos)
valores=np.array(valores)

X_train ,X_test ,Y_train,Y_test=train_test_split(granos,valores,random_state=1,test_size=0.2,stratify=valores)

primer={}
segundo={}
for i in Y_train:
  if i not in primer:
      primer[i]=1
  else:
      primer[i]+=1
    
for i in Y_test:
  if i not in segundo:
      segundo[i]=1
  else:
      segundo[i]+=1

print("conjunto de entrenamiento")
print(primer)

print("conjunto de test")
print(segundo)


def model_level():
    model=tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(64,input_shape=(27000,),activation="relu"))
    model.add(tf.keras.layers.Dense(32,activation="relu"))
    #model.add(tf.keras.layers.Dense(16,activation="relu"))
    model.add(tf.keras.layers.Dropout(rate=0.1))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model
    
    
    config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.device('/device:GPU:0'):
  model = model_level()
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
  history=model.fit(x=X_train, y=Y_train, validation_data=(X_test,Y_test), batch_size=20, epochs=10 ,verbose=1)
  scores = model.evaluate(X_test, Y_test, verbose=1,batch_size=10)
  print("Large CNN Error: {0:.4f}% acuraccy:{1:.4f}%".format((100-scores[1] * 100),scores[1]*100))
  model_json = model.to_json()
  with open("/path/granos/modelo_dense/model_dense.json","w") as json_file:
            json_file.write(model_json)
  model.save_weights('/path/granos/modelo_dense/model_dense.h5')
  
  

plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
plt.title('model accuracy-{:.2f}%'.format(scores[1]*100))
plt.legend(['Train', 'Test'], loc='upper left')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.savefig('/path/granos/modelo_dense/accuracy_dense.png')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('/path/granos/modelo_dense/loss_dense.png')
plt.show()  

from itertools import product
predictions =np.around(model.predict(X_test))
y_predict=[]

for i in predictions:
  y_predict.append(i[0])
y_predict=np.array(y_predict)
cf=confusion_matrix(Y_test,y_predict)

exp_series = pd.Series(Y_test)
pred_series = pd.Series(y_predict)

plt.imshow(cf,cmap=plt.cm.Blues,interpolation='nearest')
plt.colorbar()
plt.title('Confusion Matrix GNN')
plt.xlabel('Predicted')
plt.ylabel('Actual')
tick_marks = np.arange(len(set(exp_series))) # length of classes
class_labels = ['good','bad']
tick_marks
plt.xticks(tick_marks,class_labels)
plt.yticks(tick_marks,class_labels)
# plotting text value inside cells
thresh = cf.max() / 2.
for i,j in product(range(cf.shape[0]),range(cf.shape[1])):
    plt.text(j,i,format(cf[i,j],'d'),horizontalalignment='center',color='white' if cf[i,j] >thresh else 'black')
plt.savefig('/path/granos/modelo_dense/matrix_confusion_dense.png')
plt.show();
