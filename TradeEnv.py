import numpy as np
import torch

class ActionSpace(object):
    def __init__(self, n):
        self.n = n

    def sample(self):
        return np.random.randint(0, self.n)

class ObservationSpace(object):
    def __init__(self, shape):
        self.shape = shape

## perhaps change the action history from being time series to just a tuple of last valid trade
"""
in the stateSpace
0 index is open price difference from last tick
1 index is high price difference from last tick
2 index is low price difference from last tick
3 index is close price difference from last tick
4 index is the 200-ema difference from last tick
5 index is current close price
6 index is current 200-ema of close price
"""
class TradingEnv(object):
	def __init__(self, rawData, encodings, processedData, sequenceSize):
		self.stateSpace = None
		self.actionHistory = None
		self.encodings = encodings
		self.actionType = ["B", "S", "H"]
		self.action_space = ActionSpace(3)
		self.batchShape = (rawData.shape[0],)
		self.order_num = np.zeros(self.batchShape)
		self.createStateSpace(rawData, processedData)
		self.rawOrderPrice = None
		self.orders = np.zeros(self.batchShape)
		self.orderPrice = None
		self.hasSold = np.zeros(self.batchShape, dtype=bool)
		self.hasBought = np.zeros(self.batchShape, dtype=bool)
		self.timeIndex = 0
		self.sequenceSize = sequenceSize
		self.observation_space = ObservationSpace((sequenceSize, 8))

	def createStateSpace(self, rawData, processedData):
		self.rawStateSpace = rawData
		self.stateSpace = processedData
		self.resetActionHistory()
	
	def isTerminalState(self):
		return self.timeIndex == self.stateSpace.shape[1] - self.sequenceSize

	def isValidTrade(self, action):
		if action == "H":
			return True
		elif action == "B" and not self.hasBought:
			return True
		elif action == "S" and not self.hasSold:
			return True
		return False

	def isTradeOpen(self):
		return self.hasSold or self.hasBought

	def resetActionHistory(self):
		self.actionHistory = np.zeros(self.rawStateSpace.shape[:2] + (1,))

	def resetBuySoldIndicators(self):
		self.hasSold = np.zeros(self.batchShape, dtype=bool)
		self.hasBought = np.zeros(self.batchShape, dtype=bool)
		self.resetActionHistory()
	
	def getNextState(self):
		self.timeIndex += 1
		enc = self.encodings[:,:]
		dataHistory = self.stateSpace[:,self.timeIndex:self.timeIndex + self.sequenceSize]
		actions = self.actionHistory[:,self.timeIndex:self.timeIndex + self.sequenceSize]
		return np.concatenate([dataHistory, actions], axis=-1)

	def getRawClosePrice(self):
		return self.rawStateSpace[:,self.timeIndex + self.sequenceSize - 1, 3]

	def getClosePrice(self):
		return self.stateSpace[0,self.timeIndex + self.sequenceSize - 1, 3].item()
	
	def setActionHistory(self):
		if self.hasBought:
			self.actionHistory[0,self.timeIndex + self.sequenceSize - 1] = np.array([-self.stateSpace[0, self.timeIndex + self.sequenceSize - 1, 5]])
		elif self.hasSold:
			self.actionHistory[0,self.timeIndex + self.sequenceSize - 1] = np.array([self.stateSpace[0, self.timeIndex + self.sequenceSize - 1, 5]])
	
	def setActions(self, action):
		newHistory = self.actionHistory[:,self.timeIndex + self.sequenceSize - 1, 0]
		self.actionHistory[:,self.timeIndex + self.sequenceSize - 1, 0] = np.where(action == np.zeros(self.batchShape), 
															-self.stateSpace[:, self.timeIndex + self.sequenceSize - 1, 5], newHistory)
		newHistory = self.actionHistory[:,self.timeIndex + self.sequenceSize - 1, 0]
		self.actionHistory[:,self.timeIndex + self.sequenceSize - 1, 0] = np.where(action == np.ones(self.batchShape), 
															self.stateSpace[:, self.timeIndex + self.sequenceSize - 1, 5], newHistory)

	def step(self, action):
		if (action >= len(self.actionType)).any():
			raise Exception("Incorrect action")

		rewards = np.zeros(self.batchShape)
		#if closing sell orders
		gains = np.where(self.hasSold, 
					self.orders / self.getRawClosePrice() - self.order_num, 
					np.zeros(self.batchShape))
		rewards = np.where(action == np.zeros(self.batchShape), gains, rewards)
		orderNum = np.where(self.hasSold, np.zeros(self.batchShape), self.order_num + 1)
		self.order_num = np.where(action == np.zeros(self.batchShape),
							orderNum,
							self.order_num)
		
		orders = np.where(self.hasSold, np.zeros(self.batchShape), self.orders + self.getRawClosePrice())
		self.orders = np.where(action == np.zeros(self.batchShape), 
							orders,
							self.orders)
		#if closing buy orders
		gains = np.where(self.hasBought, 
					(self.order_num * self.getRawClosePrice() / (self.orders + 1E-20) - 1) * self.order_num, 
					np.zeros(self.batchShape))
		rewards = np.where(action == np.ones(self.batchShape), gains, rewards)
		
		orderNum = np.where(self.hasBought, np.zeros(self.batchShape), self.order_num + 1)
		self.order_num = np.where(action == np.ones(self.batchShape),
							orderNum,
							self.order_num)
		
		orders = np.where(self.hasBought, np.zeros(self.batchShape), self.orders + self.getRawClosePrice())
		self.orders = np.where(action == np.ones(self.batchShape), 
							orders,
							self.orders)

		#if holding
		rewards = np.where(action == 2 * np.ones(self.batchShape), np.zeros(self.batchShape), rewards)

		#set action history
		self.setActions(action)

		#reset indicators
		hasBought = np.logical_or(action == np.zeros(self.batchShape), self.hasBought)
		hasSold = np.logical_or(action == np.ones(self.batchShape), self.hasSold)
		self.hasBought, self.hasSold =  np.logical_and(hasBought, ~hasSold), np.logical_and(hasSold, ~hasBought)

		return self.getNextState(), rewards, self.isTerminalState(), self.order_num

	def oldStep(self, action):
		if self.actionType[action] == "B":
			if self.hasSold:
				reward = (self.orders / self.getRawClosePrice() - 1).sum().item()
				self.rawOrderPrice = None
				self.orderPrice = None
				self.resetBuySoldIndicators()
				self.orders = np.array([])
			else:
				reward = 0
				self.orders = np.concatenate([self.orders, np.array([self.getRawClosePrice()]) ])
				self.rawOrderPrice = self.getRawClosePrice()
				self.orderPrice = self.getClosePrice()
				self.hasBought = True
			self.setActionHistory()
			return self.getNextState(), reward, self.isTerminalState(), self.orders.size
		elif self.actionType[action] == "S":
			if self.hasBought:
				reward = (self.getRawClosePrice() / self.orders - 1).sum().item()
				self.rawOrderPrice = None
				self.orderPrice = None
				self.resetBuySoldIndicators()
				self.orders = np.array([])
			else:
				reward = 0
				self.orders = np.concatenate([self.orders, np.array([self.getRawClosePrice()]) ])
				self.rawOrderPrice = self.getRawClosePrice()
				self.orderPrice = self.getClosePrice()
				self.hasSold = True
			self.setActionHistory()
			return self.getNextState(), reward, self.isTerminalState(), self.orders.size
		elif self.actionType[action] == "H":
			discoutFactor = 1
			reward = 0
			if self.hasBought:
				reward = 0 #(self.getClosePrice() / self.orders - 1).sum().item() / discoutFactor
			elif self.hasSold:
				reward = 0 #(self.orders / self.getClosePrice() - 1).sum().item() / discoutFactor
			return self.getNextState(), reward, self.isTerminalState(), self.orders.size
		else:
			raise Exception(action + " is NOT a valid action")
	
	def reset(self):
		self.rawOrderPrice = None
		self.orderPrice = None
		self.hasSold = np.zeros(self.batchShape, dtype=bool)
		self.hasBought = np.zeros(self.batchShape, dtype=bool)
		self.timeIndex = np.random.randint(0, self.stateSpace.shape[1]-200)
		self.orders = np.zeros(self.batchShape)
		self.order_num = np.zeros(self.batchShape)
		self.resetActionHistory()
		enc = self.encodings[:,:]
		dataHistory = self.stateSpace[:,self.timeIndex:self.timeIndex + self.sequenceSize]
		actions = self.actionHistory[:,self.timeIndex:self.timeIndex + self.sequenceSize]
		return np.concatenate([dataHistory, actions], axis=-1) 
	
	def trueReset(self):
		self.reset()
		self.timeIndex = 0
		self.hasSold = np.zeros(self.batchShape, dtype=bool)
		self.hasBought = np.zeros(self.batchShape, dtype=bool)
		self.orders = np.zeros(self.batchShape)
		self.order_num = np.zeros(self.batchShape)
		self.resetActionHistory()
		dataHistory = self.stateSpace[:,self.timeIndex:self.timeIndex + self.sequenceSize]
		actions = self.actionHistory[:,self.timeIndex:self.timeIndex + self.sequenceSize]
		return np.concatenate([dataHistory, actions], axis=-1)

	def render(self):
		return None

