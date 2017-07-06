from keras.models import Model
from keras.layers import LSTM, Input
from keras.callbacks import LearningRateScheduler
from keras.utils.np_utils import to_categorical
from PointerLSTM import PointerLSTM
import pickle
import tsp_data as tsp
import sort_data as sort
import numpy as np


# ------------------------------------------------------------------------- #
def scheduler(epoch):
    return learning_rate
    if epoch < nb_epochs/4:
        return learning_rate
    elif epoch < nb_epochs/2:
        return learning_rate*0.5
    return learning_rate*0.1

print("preparing dataset...")

seq_len = 3
BATCH_SIZE = 1
TSP = False


if TSP:
    t = tsp.Tsp()
    X, Y = t.next_batch(BATCH_SIZE, seq_len)
    x_test, y_test = t.next_batch(2)

else:
    t = sort.DataGenerator()
    X, Y, T = t.next_batch(BATCH_SIZE, seq_len)
    #x_test, y_test, target_test = t.next_batch(2)



# t = tsp.Tsp()
# X, Y = t.next_batch(BATCH_SIZE, seq_len)
# print X
# print Y
# print "------------------------------------"
# t = sort.DataGenerator()
# X, Y, T = t.next_batch(BATCH_SIZE, seq_len)
# print X
# print Y
# print T
# exit()


YY = []
for y in Y:
    YY.append(to_categorical(y))
YY = np.asarray(YY)

hidden_size = 128
nb_epochs =  10
learning_rate = 0.3

print("building model...")
main_input = Input(shape=(seq_len, 1), name='main_input')
encoder = LSTM(output_dim = hidden_size, return_sequences = True, name="encoder")(main_input)
decoder = PointerLSTM(hidden_size, output_dim=hidden_size, name="decoder")(encoder)

model = Model(inputs=main_input, outputs=decoder)


model.compile(optimizer='adadelta',
              loss='categorical_crossentropy',
              metrics=['accuracy'])



# model.fit(X, YY, nb_epoch=nb_epochs, batch_size=64,callbacks=[LearningRateScheduler(scheduler),])
model.fit(X, T, nb_epoch=nb_epochs, batch_size=64,callbacks=[LearningRateScheduler(scheduler),])
#print(model.predict(x_test))



predictions = model.predict(X)

# print (predictions)
# exit()

pred_index = np.array([np.argmax(predictions[i],0) for i in xrange(len(predictions))])

print (pred_index)

print (Y)
exit()

print(pred_index[:5])
print("------")
print(Y[:5])
#print(to_categorical(y_test))
model.save_weights('model_weight_100.hdf5')
