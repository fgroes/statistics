import numpy as np
import matplotlib.pyplot as plt


class ExpParameters(object):
	
	def __init__(self, name):
		self.name = name

	def loadFromFile(self, fileName):
		self.fileData = []
		with open(fileName, 'r') as fid:
			for line in fid:
				self.fileData.append(line.split(';'))
		N = len(self.fileData)
		self.dataDict = {}
		for i in range(len(self.fileData[0])):
			self.dataDict[self.fileData[0][i]] = [self.fileData[j][i] for j in range(1, N)]

	def getDataArray(self, key):
		return self.dataDict[key]


def monteCarlo(p0, T, N, Nc, s):
	ct = 0
	while True:
		for i in range(N):
			r = round(normal(scale=s))
			p = p0 + r
			p %= Nbins
			dE = data[p] - data[p0]
			if dE <= 0:
				p0 = p
			else:
				pn = np.exp(- dE / T)
				if rand() < pn:
					p0 = p
				else:
					ct += 1
			if ct > Nc:
				return p0
		T *= 0.90


if __name__ == '__main__':
	freqName = 'freqSt12.txt'
	ioName = 'protokoll.csv'
	freq = np.loadtxt(freqName)
	ep = ExpParameters('Bistro')
	ep.loadFromFile(ioName)
	runs = ep.getDataArray('Run')
	runsSt12 = [run for run in runs if int(run) < 1000]
	data = -freq[0]
	rand = np.random.rand
	normal = np.random.normal	
	Nbins = len(data)
	Nss = 10000
	N = 1000
	Nc = 50
	jumpWidth = 5 
	ps = np.zeros(Nbins)
	T0 = 10 * (np.max(data) - np.min(data))
	for j in range(Nss):
		p0 = np.floor(Nbins * rand())
		p = monteCarlo(p0, T0, N, Nc, jumpWidth)
		ps[p] += 1 
	plt.plot(data / np.sum(data), 'ko')
	plt.plot(ps / np.sum(ps), 'r-')
	plt.show()
