import keras

def get_model():

    model = keras.models.Sequential()
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(units=10, return_sequences = True), input_shape = (None, 1)))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.GlobalAveragePooling1D())  # or any other pooling layer or Flatten() layer
    model.add(keras.layers.Dense(units = 1, activation = 'sigmoid'))


    opt = keras.optimizers.Adam(lr=.01)
    model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])

    return model


    #model.add(keras.layers.TimeDistributed(keras.layers.Dense(units=1, activation='sigmoid')))  # Apply Dense layer to each time step
