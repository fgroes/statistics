import numpy as np
import matplotlib.pyplot as plt


def initPlot(N):
	plt.ion()
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	ax.axis([0, N, 0, 1000])
	return ax


if __name__ == '__main__':
	x0 = 0 # m
	v0 = 20 # m s-1
	a0 = 2 # m s-2
	sx = 5 # m s-1
	sxm = 20 # m s-1
	sv = 1 # m s-2
	sa = 0.1 # m s-1
	dt = 0.1 # s
	Nm = 100
	ex = np.random.normal(0, sxm, Nm)
	ev = np.random.normal(0, sv, Nm)
	xPre = np.matrix([[x0], [v0]])
	PPre = np.matrix([[sx, 0], [0, sv]])
	F = np.matrix([[1, dt], [0, dt]])
	G = np.matrix([[dt ** 2 / 2], [dt]])
	Q = np.matrix([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]]) * sa
	H = np.matrix([1, 0])
	R = np.matrix([sxm])
	z = x0
	ax = initPlot(Nm)
	x = xPre
	P = PPre
	for i in range(Nm):   
		z = x0 + v0 * (i + 1) * dt + a0 * ((i + 1) * dt) ** 2 / 2 + ex[i]
		x = F * x + G * a0
		P = F * P * F.T + Q
		K = P * H.T * (H * P * H.T + R).I
		e = z - H * x
		x = x + K * e
		ax.plot(i, z, 'ro')
		ax.plot(i, x[0], 'ko', ms=2)
		plt.draw()
	plt.show()
