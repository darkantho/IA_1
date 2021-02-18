import scipy.io
import numpy as np
import tensorflow as tf
import hdf5storage

manolo=scipy.io.loadmat("/path/granos/buenos_manolo.mat")
datos_buenos = scipy.io.loadmat('/path/granos/grains_martian_4_all.mat')
datos_alex=hdf5storage.loadmat('/path/granos/g.mat')
indicesbuenos=hdf5storage.loadmat("/path/granos/goodGrainsIndicesAlex.mat")
buenos=[i[0] for i in indicesbuenos['goodGrainsIndicesAlex']]
indicesmalos=hdf5storage.loadmat("/path/granos/badGrainsIndicesAlex.mat")
malos=[i[0] for i in indicesmalos['badGrainsIndicesAlex']][0:len(buenos)]
num_grains = datos_buenos['grainData']['ngrains'][0][0][0][0]
level_sets = datos_buenos['grainData']['lsets']
mibatch=hdf5storage.loadmat("/path/granos/an.mat")
lista=[]
lista2=[]
'''
for i in range(2000):
  matriz=mibatch["batchAN"]["lsets"][0][i][0]
  dimension=matriz.shape
  if dimension[0]<80 and dimension[1]<80 and dimension[2]<80:
    maximo = np.max(matriz)
    matriznueva = maximo * np.ones((80, 80, 80, 1), dtype='float64')
    matriznueva[0:dimension[0], 0:dimension[1], 0:dimension[2], 0] = matriz
    lista.append(matriznueva)
'''


'''
for i in range(num_grains-100):
  matriz=level_sets[0][0][i][0]
  dimension = matriz.shape
  maximo = np.max(matriz)
  matriznueva = maximo * np.ones((80, 80, 80, 1), dtype='float64')
  matriznueva[0:dimension[0], 0:dimension[1], 0:dimension[2], 0] = matriz
  lista.append(matriznueva/maximo)
'''
'''
for i in buenos:
  matriz=datos_alex["batchAJ"][0][0][i][0]
  dimension = matriz.shape
  if dimension[0]<80 and dimension[1]<80 and dimension[2]<80:
    maximo = np.max(matriz)
    matriznueva = maximo * np.ones((80,80,80, 1), dtype='float64')
    matriznueva[0:dimension[0], 0:dimension[1], 0:dimension[2], 0] = matriz
    lista.append(matriznueva/maximo)

for i in malos:
  matriz=datos_alex["batchAJ"][0][0][i][0]
  dimension = matriz.shape
  if dimension[0]<80 and dimension[1]<80 and dimension[2]<80:
    maximo = np.max(matriz)
    matriznueva = maximo * np.ones((80,80,80,1), dtype='float64')
    matriznueva[0:dimension[0],0:dimension[1],0:dimension[2],0]=matriz
    lista.append(matriznueva/maximo)
'''
for i in range(1200):
 matriz= manolo["good"][0][0][0][i][0]
 dimension = matriz.shape
 if dimension[0]<80 and dimension[1]<80 and dimension[2]<80:
    maximo = np.max(matriz)
    matriznueva = maximo * np.ones((80,80,80,1), dtype='float64')
    matriznueva[0:dimension[0],0:dimension[1],0:dimension[2],0]=matriz
    lista.append(matriznueva/maximo)

#lista=np.array(lista)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# cargar json y crear el modelo
json_file = open('/path/granos/modelo/model.json','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = tf.keras.models.model_from_json(loaded_model_json)
# cargar pesos al nuevo modelo
loaded_model.load_weights('/path/granos/modelo/model.h5')
print("Cargado modelo desde disco.")

with tf.device('/device:GPU:0'):

  # Compilar modelo cargado y listo para usar.
  loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
  X_test=np.array(lista)
  X_test=X_test.reshape(len(lista),80,80,80,1)
  ynew = loaded_model.predict(X_test)
  buenoo=0
  maloo=0
  for i in range(len(ynew)):
      if ynew[i][0]<0.5:
        print(ynew[i][0])
        buenoo+=1
      if ynew[i][0]>0.5:
        maloo+=1
        print(ynew[i][0])



print("granos buenos:{} granos malos:{}".format(buenoo,maloo))
