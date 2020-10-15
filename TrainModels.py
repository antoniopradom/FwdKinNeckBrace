import pickle
import tensorflow as tf
import numpy as np
import pandas


def randomOrder(n):
    shu = np.arange(n)
    np.random.shuffle(shu)
    return shu


def trainNoHuman(name, maxEpochs=50, batch_size=1000, nSamples=35000, dense_layers_units=[12, 18, 32]):
    datasetPath = "./Data/"

    with open(datasetPath + 'modelData.pickle', 'rb') as f:
        modelData = pickle.load(f)

    savePath = './Output/'
    noiseTypes = ['no', 'cmb', 'in', 'md']
    dataProcessed = {}
    models = {}
    inputNames = ['q11', 'q12', 'q13']
    outputNames = ['theta1', 'theta2', 'theta3']
    metrics = []
    for nT in noiseTypes:
        dataProcessed[nT] = {}
        dataAux = modelData.where(modelData['NoiseType'] == nT).dropna()
        randOrder = randomOrder(dataAux.shape[0])[:nSamples]
        x = dataAux[inputNames].values[randOrder]
        y = dataAux[outputNames].values[randOrder]
        models[nT] = tf.keras.Sequential([tf.keras.layers.Dense(6, activation='relu', input_shape=(len(inputNames),))])
        # if we put the constructor in a for loop, we can easily increase the number of layers
        for n in dense_layers_units:
            models[nT].add(tf.keras.layers.Dense(n, activation='relu'))
        # the last layer reshapes the tensor to the required output size
        models[nT].add(tf.keras.layers.Dense(len(outputNames)))
        # we use the built-in optimizer RMSprop. Using the tf.keras.optimizers class, we can easily change optimizers
        optimizer = tf.keras.optimizers.Adam(0.001)
        # in here we declare the training parameters of the network
        models[nT].compile(loss='mse',  # mean square error
                           optimizer=optimizer,
                           metrics=['mae', 'mse'])
        history = models[nT].fit(x, y, batch_size=batch_size, epochs=maxEpochs, validation_split=0.2)
        models[nT].save(savePath + nT + '.h5')
        # in here we can see the training history and plot it
        dataProcessed[nT]['histogram'] = pandas.DataFrame(history.history)

        # metrics.append(auxAux.copy())

    dataProcessed['metrics'] = pandas.concat(metrics)
    with open(savePath + 'processedData_%s.pickle' % name, 'wb') as f:
        pickle.dump(dataProcessed, f)
