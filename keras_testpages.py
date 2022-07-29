import numpy
import matplotlib.pyplot as plt
import pandas
import math
import csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# test your packs

# fix random seed for reproducibility
numpy.random.seed(24)

# .txt trans to .csv
# csvFile = open("lstm_data.csv", 'w', newline='')
# writer = csv.writer(csvFile, dialect='excel')
# f = open("test.txt", "r")
# for line in f.readlines():
#     line = line.replace('|', '\t')
#     csvRow = line.split()
#     writer.writerow(csvRow)

# extra used column from csv
# data = pandas.read_csv('lstm_data.csv', header=None)
# data.columns = ["date", "org_price", "final_price", "cnt"]
# data.to_csv('used_data.csv', columns={"date", "cnt"}, index=False)


# load the dataset
dataframe = pandas.read_csv('used_data.csv', usecols=[0], engine='python', skipfooter=3)
dataset = dataframe.values
dataset = dataset.astype('float32')

#normalizing
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# split into train and test sets
#67% train 33% test
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]       # 保持原有数据序列
print(len(train), len(test))


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)

look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)



# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))       # keras LSTM模型对数据形式要求为3D tensor
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
#model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

#save model loss
hist = model.fit(trainX, trainY, epochs=5, batch_size=1, verbose=2)
loss = hist.history
with open('C:\\Users\\h50024124\\Desktop\\LSTM_Predict\\history_loss.txt','w')as f:
    f.write(str(loss))
# serialize model to JSON
model_json = model.to_json()
with open("C:\\Users\\h50024124\\Desktop\\LSTM_Predict\\model.json", "w") as json_file:
    json_file.write(model_json)
#serialize weights to HDF5
model.save_weights("C:\\Users\\h50024124\\Desktop\\LSTM_Predict\\model.h5")

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset), label='origin_data')
plt.plot(trainPredictPlot, label='train_predict')
plt.plot(testPredictPlot, label='test_predict')
plt.legend(loc=0)
plt.show()
