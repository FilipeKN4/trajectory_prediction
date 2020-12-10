import numpy as np

# bibliotecas para uso de redes neurais. Neste caso, do tipo LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# bibliotecas para o pré-processamento dos dados. Normalização, etc...
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error

def dataset_with_look_back_for_multiple_predictors(dataset, look_back=1, initial_column=1, last_column=2, target_column=1):
    dataX, dataY = [], []
    
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), initial_column:last_column]
        dataX.append(a)
        dataY.append(dataset[i + look_back, target_column])
        
    return np.array(dataX), np.array(dataY)

def lstm(train_x, train_y, reps=1):
    model = Sequential()

    # camada LSTM    
    # units é a quantidade de células de memória
    # return_sequences se for True retorna os valores de saída para a próxima camada
    # input_shape recebe (timesteps, input_dim)
    model.add(LSTM(units=256, input_shape=(train_x.shape[1], train_x.shape[2])))
    # O dropout é usado para evitar overfitting
    model.add(Dropout(0.3))

    # camada Dense 1
    model.add(Dense(units=256, activation='tanh'))
    model.add(Dropout(0.3))

    # camada Dense 2
    model.add(Dense(units=128, activation='tanh'))
    model.add(Dropout(0.3))

    # camada final com units de 2 para duas saídas
    # activation define a função de ativação para o cálculo da última saída
    model.add(Dense(units=2, activation='tanh'))

    # optimizer é a função utilizada para o cálculo do gradiente
    # loss é o erro utilizado para o ajuste dos pesos
    model.compile(optimizer="rmsprop", loss="mean_squared_error", 
                metrics=["mean_absolute_error"] )

    # método para parada de treinamento antes do número de epochs definido
    # monitor indica a função a ser verificada a melhora
    # min_delta indica o valor mínimo de melhora aceito para não parar o treinamento
    # patience indica o número de iterações do treino necessários com min_delta abaixo do esperado
    # es = EarlyStopping(monitor='loss', min_delta=0.01, patience=10, verbose=1)

    # epochs é a quantidade de vezes que ocorrerá o ajuste dos pesos
    # batch size é o número de amostras por atualização de gradiente
    for rep in range(reps):
        model.fit(train_x, train_y, epochs=5, batch_size = 32)
        model.summary()

    return model