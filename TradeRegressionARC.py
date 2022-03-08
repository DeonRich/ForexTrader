import math
import random
import numpy as np
import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
#from TradeEnv import TradingEnv
from collections import namedtuple
CLOSE_INDEX = 3
EMA_INDEX = 4
SEQUENCE_FEATURES = 5
SEQUENCE_LEN = 100
ENCODING_LEN = 12
TOTAL_EPOCHS = 3
LEARNING_RATE = .001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
saveLocation = "TrainingInfo/"
class ForexDataset(Dataset):
    def __init__(self, X, encodings, Y, std, mean):
        """
        Args:
            X
            Y
        """
        self.X = (X - mean) / std
        self.Y = (Y - mean) / std
        self.X = self.X[:,:,:]
        self.encodings = encodings

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.X[idx], self.encodings[idx], self.Y[idx]
    
    def print_info(self, info=""):
    	print(f"{info} X: {self.X.size()}, encX: {self.encodings.size()} Y: {self.Y.size()}")

class DQN(nn.Module):
	def __init__(self):
		super(DQN, self).__init__()
		self.num_layers = 1
		self.hidden_size = 32
		self.sequence_size =  SEQUENCE_LEN
		self.rnn1 = nn.RNN(SEQUENCE_FEATURES, self.hidden_size, self.num_layers, batch_first =True)
		self.fc1 = nn.Linear(self.hidden_size * self.sequence_size + ENCODING_LEN, 32)
		self.fc2 = nn.Linear(32, 1)
		#self.fc3 = nn.Linear(48, 1)

	def forward(self, x, encoding):
		h0 =  torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
		#c0 =  torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
		x, _ = self.rnn1(x, h0)
		x = F.relu(x).reshape(x.shape[0], -1)
		x = torch.cat([x, encoding], axis=1)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		#x = self.fc3(x)
		return x
def currencyToTwoHot(name):
	convert = ["AUD", "CAD", "CHF", "JPY", "NZD", "USD", "EUR", "GBP", "HKD", "CNH", "MXN", "XAU"]
	encoding = [0 for i in range(len(convert))]
	encoding[convert.index(name[0:3])] = 1
	encoding[convert.index(name[3:6])] = 1
	return encoding

def obtainAndPadData(path = "C:\\Users\\Deon\\Documents\\CS230\\Data"):
	for root, dirs, files in os.walk(path, "r"):
		eachCurrency = []
		currencyEncoding =[]
		maxLen = 0
		for name in files:
			file = open(os.path.join(root, name))
			result = []
			encoding = currencyToTwoHot(name)
			currencyEncoding.append(encoding)
			for line in file.readlines():
				result.append(line.split(",")[2:6])
			eachCurrency.append(torch.from_numpy(np.array(result, dtype = 'float')))
			maxLen = max(eachCurrency[-1].shape[0], maxLen)
			print(f"{name}: {eachCurrency[-1].size()}")
			file.close()
	print(f"number of currencies: {len(eachCurrency)}")
	newCurrency = []
	for each in eachCurrency:
		currentLen = each.size()[0]
		newCurrency.append(torch.unsqueeze(F.pad(each, (0,0,0,maxLen-currentLen)), 0))
	eachCurrency = torch.cat(newCurrency)
	print(f"size: {eachCurrency.size()}")
	return eachCurrency, torch.tensor(currencyEncoding)

def createTrainingData(rawData, encodings, emas):
	sequenceLen = 100
	devSetSize = 2000
	X = []
	Y = []
	emas = emas.unsqueeze(2)
	encodingInput = []
	for m in range(rawData.size(0)):
		for i in range(rawData.size(1) - sequenceLen - 1):
			encodingInput.append(encodings[m:m+1, :])
			X.append(torch.cat([rawData[m:m+1, i:i+sequenceLen, :], emas[m:m+1, i:i+sequenceLen, :]], axis=-1))
			Y.append(rawData[m:m+1, i+sequenceLen+1, CLOSE_INDEX:CLOSE_INDEX+1])
	
	indices = torch.load(saveLocation+"shuffled_indices.pt").tolist()
	#indices = np.arange(len(X))
	#np.random.shuffle(indices)
	#torch.save(rawData, saveLocation+"rawData.pt")
	#torch.save(torch.from_numpy(indices),  saveLocation+"shuffled_indices.pt")
	X = torch.cat(X)
	Y = torch.cat(Y)
	encodingInput = torch.cat(encodingInput)
	X = X[indices]
	Y = Y[indices]
	encodingInput[indices]
	trainingX, testX = X[:-devSetSize], X[-devSetSize:]
	trainingY, testY = Y[:-devSetSize], Y[-devSetSize:]
	train_encX, test_encX = encodingInput[:-devSetSize], encodingInput[-devSetSize:]
	#print(f"Training set size: {trainingX.size()}, label size: {trainingY.size()}")
	#print(f"Encoding set size: {train_encX.size()}, test size: {test_encX.size()}")
	#print(f"Test set size: {testX.size()}, label size: {testY.size()}")
	return trainingX, trainingY, testX, testY, train_encX, test_encX

def calculateEMA(rawData):
	ema_window = 200
	beta = 1 - 1/ema_window
	corrected_ema = [rawData[:,0,CLOSE_INDEX:CLOSE_INDEX+1]]
	ema = (1-beta) * corrected_ema[-1]
	for i in range(1,rawData.size(1)):
		ema = beta * ema + (1-beta) * rawData[:,i,CLOSE_INDEX:CLOSE_INDEX+1]
		corrected_ema.append(ema / (1 - beta**(i+1)))
	corrected_ema = torch.cat(corrected_ema, axis=1)
	return corrected_ema

def evaluateModel(test_net, x_enc, BaseWriteName):
	sigma, mu = torch.load(saveLocation+"std.pt"), torch.load(saveLocation+"mean.pt")
	rawX, raw_enc = x_enc
	rawX = ((rawX - mu) / sigma)
	for i in range(rawX.size(0)):
		print(f"Evaluating data:{i} and saving figure predictions...")
		preds = rawX[i, :SEQUENCE_LEN, CLOSE_INDEX].tolist()
		actual_closing = rawX[i, : , CLOSE_INDEX].tolist()
		enc_x = raw_enc[i:i+1, :]
		for start in range(rawX.size(1) - SEQUENCE_LEN - 1):
			inputs = rawX[i:i+1, start:start+SEQUENCE_LEN, :].float().to(device)
			out = test_net(inputs, enc_x)
			preds.append(out.item())
		plt.clf()
		plt.title("Closing Price over time")
		plt.plot(preds, label = "Predicted")
		plt.plot(actual_closing, label = "Actual")
		plt.plot(rawX[i,:,EMA_INDEX].tolist(), label = "200-Ema")
		plt.legend()
		plt.ylabel("MSE Loss")
		plt.xlabel("Epochs")
		plt.savefig(f"{BaseWriteName}{i}_Closing_Price_Plot.png")	

def train_regression():
	std, mean = torch.load(saveLocation+"std.pt"), torch.load(saveLocation+"mean.pt")
	print(f"std:{std.item()} mean:{mean.item()}")
	
	trainSet = ForexDataset(torch.load(saveLocation+"trainX.pt"), 
									torch.load(saveLocation+"train_encX.pt"),
									torch.load(saveLocation+"trainY.pt"), std, mean)
	testSet = ForexDataset(torch.load(saveLocation+"testX.pt"), 
									torch.load(saveLocation+"test_encX.pt"),
									torch.load(saveLocation+"testY.pt"), std, mean)
	trainSet.print_info("TrainSet:")
	testSet.print_info("TestSet:")

	trainloader = torch.utils.data.DataLoader(trainSet, batch_size=100,
	                                          shuffle=True)
	testloader = torch.utils.data.DataLoader(testSet, batch_size=100,
	                                         shuffle=False)
	net = DQN().to(device)

	criterion = nn.MSELoss()
	optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))
	
	loss_history = []
	val_loss_history = []
	for epoch in range(TOTAL_EPOCHS):  # loop over the dataset multiple times
		running_loss = 0.0
		epoch_loss = 0.0
		val_epoch_loss = 0.0
		for i, data in enumerate(trainloader, 0):
			# get the inputs; data is a list of [inputs, labels]
			inputs, encodings, labels = data[0].float().to(device), data[1].float().to(device), data[2].float().to(device)
			# zero the parameter gradients
			optimizer.zero_grad()

			# forward + backward + optimize
			outputs = net(inputs, encodings)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
			# print statistics
			running_loss += loss.item()
			epoch_loss += loss.item()
			if i % 10 == 9:    # print every 10 mini-batches
				print('[%d, %5d] loss: %.7f' %
	                  (epoch + 1, i + 1, running_loss / 10))
				running_loss = 0.0
		loss_history.append(epoch_loss / (i+1))

		running_loss = 0.0
		for i, data in enumerate(testloader, 0):
			# get the inputs; data is a list of [inputs, labels]
			inputs, encodings, labels = data[0].float().to(device), data[1].float().to(device), data[2].float().to(device)
				
			# forward + backward + optimize
			outputs = net(inputs, encodings)
			loss = criterion(outputs, labels)
			# print statistics
			running_loss += loss.item()
			val_epoch_loss += loss.item()
			if i % 10 == 9:    # print every 10 mini-batches
				print('[%d, %5d] val_loss: %.7f' %
				      (epoch + 1, i + 1, running_loss / 10))
				running_loss = 0.0
		val_loss_history.append(val_epoch_loss / (i+1))
	return net, loss_history, val_loss_history
"""rawData, encodings = obtainAndPadData()
trainX, trainY, testX, testY, train_encX, test_encX = createTrainingData(torch.load(saveLocation+"rawData.pt"),
																			torch.load(saveLocation+"rawEncodings.pt"),
																			calculateEMA(torch.load(saveLocation+"rawData.pt")))

torch.save(trainX, saveLocation + "trainX.pt")
torch.save(trainY, saveLocation + "trainY.pt")
torch.save(testX, saveLocation + "testX.pt")
torch.save(testY, saveLocation + "testY.pt")"""

#trainX, trainY, testX, testY, train_encX, test_encX = createTrainingData(torch.load(saveLocation+"rawData.pt"), torch.load(saveLocation+"rawEncodings.pt"))
#torch.save(train_encX, saveLocation + "train_encX.pt")
#torch.save(test_encX, saveLocation + "test_encX.pt")


if __name__ == "__main__":
	print(f"Running on device: {device}")
	#train model
	net, loss_history, val_loss_history = train_regression()
	
	# Save Model
	baseWritePath = saveLocation + "Model_"
	model_count = 0
	while os.path.exists(baseWritePath + str(model_count) + "/"):
		model_count += 1
	modelWritePath = baseWritePath + str(model_count) + "/"
	os.mkdir(modelWritePath)
	torch.save(net.state_dict(), f"{modelWritePath}Model_{model_count}.pt")

	#Evaluate model on all training/dev data
	rawX = torch.load(saveLocation + "rawData.pt")
	emas = calculateEMA(rawX).unsqueeze(2)
	rawX = torch.cat([rawX, emas], axis=-1)		
	raw_enc = torch.load(saveLocation + "rawEncodings.pt").float().to(device)
	data = (rawX, raw_enc)
	evaluateModel(net, data, modelWritePath)

	#Evaluate model on unseen test data
	rawX = torch.load(saveLocation + "rawUnseenData.pt")
	emas = calculateEMA(rawX).unsqueeze(2)
	rawX = torch.cat([rawX, emas], axis=-1)		
	raw_enc = torch.load(saveLocation + "rawUnseenEnc.pt").float().to(device)
	data = (rawX, raw_enc)
	evaluateModel(net, data, modelWritePath+"Test")	

	plt.clf()
	plt.plot(loss_history, label = "Train")
	plt.plot(val_loss_history, label = "Test")
	plt.legend()
	plt.ylabel("MSE Loss")
	plt.xlabel("Epochs")
	plt.savefig(f"{modelWritePath}Loss.png")
	plt.show()