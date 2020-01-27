from numpy import array
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense

def split_sequence(dizi,adim_sayisi):
    girdiler,hedef = list(),list()
    for i in range(len(dizi)):
        end_ix = i + adim_sayisi
        if end_ix > len(dizi) - 1:
            break

        seq_x , seq_y = dizi[i:end_ix],dizi[end_ix]
        girdiler.append(seq_x)
        hedef.append(seq_y)

    return array(girdiler),array(hedef)

dizi = [1,2,3,4,5,6,7,8,9]

adim_sayisi = 3

girdiler,hedef = split_sequence(dizi,adim_sayisi)

for i in range(len(girdiler)):
    print(girdiler[i],hedef[i])

model = Sequential()
model.add(Dense(100,activation="relu",input_dim=adim_sayisi))
model.add(Dense(1))
model.compile(optimizer= "adam",loss = "mae")

history = model.fit(girdiler,hedef,validation_split=0.2,epochs=1000,verbose=0)

loss = history.history["loss"]
epochs = range(1,len(loss)+1)

plt.figure()
plt.plot(epochs,loss,'bo',label= 'Training loss')
plt.plot(epochs, loss,'b',label= 'Validaion loss')
plt.title("Training and Validation Loss")
plt.legend()
plt.show()

yeni_girdi = array([7,8,9])
yeni_girdi = yeni_girdi.reshape(((1,adim_sayisi)))
tahminler = model.predict(yeni_girdi)
print(tahminler)