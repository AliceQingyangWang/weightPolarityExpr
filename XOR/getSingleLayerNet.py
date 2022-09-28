import keras
def getSingleLayerNet(inputSize, initializer, hiddenSize = 64):
    return keras.models.Sequential([
        keras.layers.Input(shape=(inputSize, )),
        keras.layers.Dense(hiddenSize, activation='relu', kernel_initializer = initializer, bias_initializer = keras.initializers.GlorotUniform(), name='dense_1'),
        # keras.layers.Dense(512, activation='relu', kernel_initializer = initializer, bias_initializer = keras.initializers.Zeros(), name='dense_1', kernel_constraint=keras.constraints.non_neg()),
        # keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation='sigmoid', kernel_initializer = initializer, bias_initializer = keras.initializers.GlorotUniform(),name='dense_2')
        # keras.layers.Dense(numClases, activation='softmax', kernel_initializer = initializer, bias_initializer = keras.initializers.Zeros(),name='dense_2', kernel_constraint=keras.constraints.non_neg())
    ])