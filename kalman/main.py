import numpy as np
import matplotlib.pyplot as plt


def initPlot(N):
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	ax.axis([0, N, 0, 500])
	return ax


class filterKalman(object):

	def __init__(self):
		self.optimal = False
	
	def setModelStateTransition(self, modelStateTransition):
		self.F = np.matrix(modelStateTransition)

	def setModelControlInput(self, modelControlInput):
		self.G = np.matrix(modelControlInput)

	def setParameterControlInput(self, parameterControlInput):
		self.u = np.matrix(parameterControlInput)

	def setNoiseProcess(self, noiseProcess):
		self.Q = np.matrix(noiseProcess)

	def setModelObservation(self, modelObservation):
		self.H = np.matrix(modelObservation)

	def setNoiseObservation(self, noiseObservation):
		self.R = np.matrix(noiseObservation)

	def setState(self, state, covariance=None):
		self.x = np.matrix(state)
		N = len(self.x)
		if covariance == None:
			self.P = np.matrix(zeros((N, N)))
		else:
			self.P = np.matrix(covariance)
		self.E = np.matrix(np.diag(np.ones(N)))

	def __predict(self):
		self.x = self.F * self.x + self.G * self.u
		self.P = self.F * self.P * self.F.T + self.Q

	def setOptimalKalmanGain(self, optimal):
		self.optimal = optimal

	def __update(self):
		K = self.P * self.H.T * (self.H * self.P * self.H.T + self.R).I
		e = self.z - self.H * self.x
		self.x = self.x + K * e
		if self.optimal == True:
			self.P = (self.E - K * self.H) * self.P
		else:
			KH = self.E - K * self.H
			self.P = KH * self.P * KH.T + K * self.R * K.T

	def setObservation(self, observation):
		self.z = observation

	def advance(self, observation):
		self.z = observation 
		self.__predict()
		self.__update()

	def getState(self):
		return self.x


if __name__ == '__main__':
	x0 = 0 # m
	v0 = 20 # m s-1
	a0 = 2 # m s-2
	sx = 5 # m s-1
	sxm = 4 # m s-1
	sv = 1 # m s-2
	sa = 0.1 # m s-1
	dt = 0.1 # s
	Nm = 100
	kalman = filterKalman()
	kalman.setModelStateTransition([[1, dt], [0, dt]])
	kalman.setModelControlInput([[dt ** 2 / 2], [dt]])
	kalman.setParameterControlInput(a0)
	kalman.setNoiseProcess(sa * np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]]))
	kalman.setModelObservation([1, 0])
	kalman.setNoiseObservation([sxm])
	kalman.setState([[x0], [v0]], [[sx, 0], [0, sv]])
	ex = np.random.normal(0, sxm, Nm)
	z = np.array([x0 + v0 * (i + 1) * dt + a0 * ((i + 1) * dt) ** 2 / 2 + ex[i] for i in range(Nm)])
	ax = initPlot(Nm)
	xx = np.zeros((2, Nm))
	#kalman.setOptimalKalmanGain(True)
	for i in range(Nm):   
		kalman.advance(z[i])
		xx[:, i] = kalman.getState().T
	ax.plot(z, 'ko')
	ax.plot(xx[0, :], 'r-', lw=2)
	plt.show()
