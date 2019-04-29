#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import vegas, math

CA = 3.
CF = 4./3.
alphas = 0.3
T = 0.4
p0 = 16
mD2 = 6*np.pi*alphas*T**2
omega = 3

x = omega/p0
qmax = p0
colors = (CA+CA*(1-x)**2 + (2*CF-CA)*x**2)/2.
qhat0 = alphas*colors*mD2*T
qhat = alphas*colors*mD2*T*np.log(1.+np.sqrt(2*x*(1-x)*p0*qhat0)/mD2)
tauf = np.sqrt(2*x*(1-x)*p0/qhat)
def P(x):
	return CF*(1+(1-x)**2)/x

def C(q2):
	return 4*np.pi*alphas*T*mD2/(q2*(q2+mD2))
	
def deltaE(pT2, x):
	return (pT2+(1-x)*mD2/2.)/(2.*x*(1-x)*p0)
	
def VecPsi(px, py, x, t):
	dE = deltaE(px**2+py**2, x)
	return np.array([px/dE, py/dE])
	
integ = vegas.Integrator([[-qmax, qmax],[-qmax, qmax],[-qmax, qmax],[-qmax, qmax]])

def get_at(x,t):
	def Kernel(vector):
		px, py, qx, qy = vector
		q2 = qx**2 + qy**2
		dE = deltaE(px**2+py**2, x)
		A0x, A0y = px/dE, py/dE
		Ax, Ay = VecPsi(px, py, x, t) - VecPsi(px-qx, py-qy, x, t) 
		Bx, By = VecPsi(px, py, x, t) - VecPsi(px+x*qx, py+x*qy, x, t) 
		Cx, Cy = VecPsi(px, py, x, t) - VecPsi(px+(1-x)*qx, py+(1-x)*qy, x, t) 
		res = A0x * C(q2) * (CA/2.*Ax + (2*CF-CA)/2.*Bx + CA/2.*Cx) \
			+ A0y * C(q2) * (CA/2.*Ay + (2*CF-CA)/2.*By + CA/2.*Cy)
		return res*(1.-np.cos(dE*t))
	
	result = integ(Kernel, nitn=10, neval=1000)
	return result.mean

def Gamma(x, t):
	return alphas * P(x) * get_at(x,t) / (x*(1.-x))**2 / p0**3 / (2*np.pi)**4
	

for t in np.linspace(0.1,10,50):
	plt.plot(t/5.026, Gamma(x, t),'ro')
final = Gamma(x, 25)
plt.plot([tauf/5.026, tauf/5.026], [0, final],'k--')
plt.show()

