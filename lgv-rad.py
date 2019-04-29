#!/usr/bin/env python3
from scipy.special import sici
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

alphas = 0.3
T = 0.2
CA = 3
CF = 4/3
mD2 = 6*np.pi*alphas*T**2
qhat = CA*mD2*alphas*T
omegac = np.pi*T

def f(x):
	si, ci = sici(x)                                  
	s = np.sin(x)
	c = np.cos(x)
	return -(-ci*x**3+3*x**2*si + x**2*s - 3*x + s + 2*x*c)/3./x**3


def Prob(E, dt):
	return alphas*CF*qhat*dt**3 * (f(dt*E/2) - f(dt*omegac/2))

def lambdaT(E):
	def eq(dt):
		return Prob(E, dt) - 1.0
	res = brentq(eq, 0.1/T, 10./T)
	print(res*T)

E = np.linspace(5,200,200)

plt.plot(E, Prob(E, lambdaT(E)))
plt.show()
