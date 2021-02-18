import numpy as np
import scipy.io
from sklearn.preprocessing import StandardScaler

lista_buenos=[]
lista_malos=[]
#granos_manolo_buenos=scipy.io.loadmat('/path/good.mat')
#granos_manolo_buenos2=granos_manolo_buenos["gG"]
granos_manolo_malos=scipy.io.loadmat('/path/granos/bad.mat')
granos_manolo_malos2=granos_manolo_malos["bG"]
granos_sebas_buenos=scipy.io.loadmat('/path/granos/BATCH-SP.mat')
granos_sebas_buenos2=granos_sebas_buenos["lsets"]
granos_malos_letechi_op=scipy.io.loadmat('/path/granos/malos_letechi.mat')
granos_malos_letechi=granos_malos_letechi_op["malos"]
granos_buenos_marcianos=scipy.io.loadmat('/path/granos/grains_martian_4_all.mat')
granos_buenos_marcianos2=granos_buenos_marcianos["grainData"]["lsets"]

# dimensionesmalos=[]
# dimensionesbuenos=[]

# for i in range(646) :
#   dimension=granos_manolo_malos2[i][0].shape
#   dimensionesmalos.append(dimension)

# for i in range(1643):
#   dimension=granos_sebas_buenos2[i][0].shape
#   dimensionesbuenos.append(dimension)

# for i in range(1677):
#       dimension=granos_malos_letechi[i][0].shape      
#       dimensionesmalos.append(dimension)

# f=open("/path/dimensiones_granos.csv","w")

# f.write("granos malos;;\n")
# for i in dimensionesmalos:
#   linea=map(str,list(i))
#   f.write("{}\n".format(";".join(linea)))
# f.write("\n")
# f.write("granos buenos;;\n")

# for i in dimensionesbuenos:
#   linea=map(str,list(i))
#   f.write("{}\n".format(";".join(linea)))
# f.close()

# for i in range(granos_buenos_marcianos["grainData"]["ngrains"][0][0][0][0]):  
#       dimension=granos_buenos_marcianos2[0][0][i][0].shape
#       dimensiones.append(dimension)

# axis_x=[]
# axis_y=[]
# axis_z=[]

# for i in dimensiones:
#   axis_x.append(i[0])
#   axis_y.append(i[1])
#   axis_z.append(i[2])


# print("medias x:{} media y:{} media z:{}".format(np.mean(np.array(axis_x)),np.mean(np.array(axis_y)),np.mean(np.array(axis_z))))    
# print("desviacion x:{} desviacion y:{} desviacion z:{}".format(np.std(np.array(axis_x)),np.std(np.array(axis_y)),np.std(np.array(axis_z))))
# print("mediana x:{} mediana y:{} mediana z:{}".format(np.median(np.array(axis_x)),np.median(np.array(axis_y)),np.median(np.array(axis_z))))




for i in range(646) :
  dimension=granos_manolo_malos2[i][0].shape
  if dimension[0]<30 and dimension[1]<30 and dimension[2]<30:
    valor=granos_manolo_malos2[i][0]
    maximo=np.max(valor)
    matriz=maximo*np.ones((30,30,30),dtype="float64")
    matriz[0:dimension[0],0:dimension[1],0:dimension[2]]=valor
    lista_malos.append(np.ravel(matriz))

    
for i in range(1643):
  dimension=granos_sebas_buenos2[i][0].shape
  if dimension[0]<30 and dimension[1]<30 and dimension[2]<30:
     valor=granos_sebas_buenos2[i][0]
     maximo=np.max(valor)
     matriz=maximo*np.ones((30,30,30),dtype="float64")
     matriz[0:dimension[0],0:dimension[1],0:dimension[2]]=valor
     lista_buenos.append(np.ravel(matriz))


for i in range(1677):
      dimension=granos_malos_letechi[i][0].shape      
      if dimension[0]<30 and dimension[1]<30 and dimension[2]<30:
          valor=granos_malos_letechi[i][0]
          maximo=np.max(valor)
          matriz=maximo*np.ones((30,30,30),dtype="float64")
          matriz[0:dimension[0],0:dimension[1],0:dimension[2]]=valor
          lista_malos.append(np.ravel(matriz))


for i in range(granos_buenos_marcianos["grainData"]["ngrains"][0][0][0][0]):
      if len(lista_buenos)!=len(lista_malos):
           dimension=granos_buenos_marcianos2[0][0][i][0].shape
           if dimension[0]<30 and dimension[1]<30 and dimension[2]<30:
                   valor=granos_buenos_marcianos2[0][0][i][0]
                   maximo=np.max(valor)
                   matriz=maximo*np.ones((30,30,30),dtype="float64")
                   matriz[0:dimension[0],0:dimension[1],0:dimension[2]]=valor
                   lista_buenos.append(np.ravel(matriz))

print(len(lista_buenos))
print(len(lista_malos))   

lista_buenos=np.array(lista_buenos)
lista_malos=np.array(lista_malos)
buenos=StandardScaler()
malos=StandardScaler()
buenos.fit(lista_buenos)
malos.fit(lista_malos)
buenos_std=buenos.transform(lista_buenos)
malos_std=malos.transform(lista_malos)

headers=[]
for i in range(27000):
   headers.append("X{}".format(i))
headers.append("Label")
f=open("/path/granos/granos_train_test.csv","w")
f.write("{}\n".format(";".join(headers)))
for i in buenos_std:
     i=list(i)
     i.append(0)
     linea=map(str,i)
     f.write("{}\n".format(";".join(linea)))

for i in malos_std:
     i=list(i)
     i.append(1)
     linea=map(str,i)
     f.write("{}\n".format(";".join(linea)))


f.close()


import pandas as pd

df=pd.read_csv("/path/granos_train_test.csv",sep=";")
df.head()
