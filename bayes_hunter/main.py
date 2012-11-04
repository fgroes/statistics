import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


class Hypothesis(object):
	
	def __init__(self, name):
		self.name = name

	def setProbability(self, probability):
		self.prob = probability

	def getProbability(self):
		return self.prob

	def getMaxIdx(self):
		idx = np.argmax(self.prob)
		return idx / N, idx % N


class HypothesisPosition(Hypothesis):
	
	def setX(self, x):
		self.x = x

	def getX(self):
		return self.x

	def setY(self, y):
		self.y = y

	def getY(self):
		return self.y

	def getPosOfMax(self):
		n, m = self.getMaxIdx()
		return self.x[n, m], self.y[n, m]


def simDataNormal(x0, y0, sigma, N):
	x = np.random.normal(x0, sigma, N)
	y = np.random.normal(y0, sigma, N)
	return x, y


def likelihood(x, y, sigma, hypothesis):
	return 1.0 / (np.sqrt(2 * np.pi) * sigma) \
		* np.exp(-((hypothesis.getX() - x) ** 2 + (hypothesis.getY() - y) ** 2) / (2 * sigma ** 2))


def initFigure():
	plt.ion()
	fig = plt.figure()
	gs = mpl.gridspec.GridSpec(3, 1, height_ratios=[4, 1, 1])
	axIm = fig.add_subplot(gs[0])
	axX = fig.add_subplot(gs[1])
	axY = fig.add_subplot(gs[2])
	axX.axis([0, Nu, 0, s])
	axY.axis([0, Nu, 0, s])
	axX.plot(range(0, Nu), np.ones(Nu) * x0, 'r-')
	axY.plot(range(0, Nu), np.ones(Nu) * y0, 'r-')
	return fig, axIm, axX, axY


if __name__ == '__main__':
	sigma = 200
	sigmaEst = 100
	x0 = 200
	y0 = 500
	s = 1000
	N = 100
	xr = np.linspace(0, s, N)
	xx, yy = np.meshgrid(xr, xr)
	hyp = HypothesisPosition('object at position')
	hyp.setX(xx)
	hyp.setY(yy)
	hyp.setProbability(np.ones([N, N]) / (N ** 2))
	Nu = 100
	x, y = simDataNormal(x0, y0, sigmaEst, Nu)
	x[0] = 900
	y[0] = 100
	fig, axIm, axX, axY = initFigure()
	for i in range(Nu):
		liklh = likelihood(x[i], y[i], sigma, hyp)
		posterior = hyp.getProbability() * liklh
		posterior /= np.sum(posterior)
		hyp.setProbability(posterior)
		xm, ym = hyp.getPosOfMax()
		axX.plot(i, xm, 'ko', ms=2)
		axY.plot(i, ym, 'ko', ms=2)
		axIm.cla()
		axIm.imshow(hyp.getProbability())
		plt.draw()
	plt.show()
