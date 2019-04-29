#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import vegas, math
from scipy import interpolate, integrate

# color factors
CA = 3.
CF = 4./3.
# coupling
alphas = 0.3
g2 = 4*np.pi*alphas
# splitting kinematics
E = 16
omega = 3.0
x = omega/E
M = x*(1-x)*E # not to be confused with the mass
# temperature, screening mass, effective splitting mass, regulator mass
T = .2
mD2 = 6*np.pi*alphas*T**2
meff2 = (1-x+CF/CA*x**2)*mD2/2.
m2regulator = 1e-8*meff2
# a crude estimate of qhat (not really needed)
qhat0 = alphas*CA*mD2*T
# upper limit of the q2 table, make sure this is much greater than sqrt(M*qhat0) and mD2
qmax2 = 100*np.sqrt(2*x*(1-x)*qhat0*E)#400*mD2
print(qmax2)
# time units
tau0 = 5 #np.sqrt(2*M/qhat0)
# time steps
dt = tau0*0.1


class LO_branching:
	"""
	LO branching rate for q --> q + g: 
	The method is described in the appendix of 1006.2379 by Caron-Huot and Gale
	
	This solver solves the time evoltuion of the vecotr valued wave funvtion vecPsi
	By rotation symmetry, Psi is decomposed into:
		vecPsi(px, py) = [px, py]/(p^2 + meff^2)^2 Phi(p^2)
	This class actually evolves the time evolution of Phi(p^2), with initial condition
		Phi(p^2, t=0) = C[i*(p^2 + meff^2)/2/M]
	The time evolution is:
		partial Phi(p^2, t)
		--------------------- =  - C[Phi(t)]
		partial t
	Where the C operator is a integral operator in the interaction picture. 
	To speed up the calculation, Phi(p^2) values are cubic interpolated (and 
	linearly extrapolated) from a finite number of grids equally spaced in ln(p2/mD2)
	
	Once the time evolution is finished, the rate is calcualted as:
		R(t1) = N(t1)/E /(2\pi)^2 * g^2 * CF * (1+(1-x)^2)/x
		N(t1) = int dp2 int_0^t1 dt p^2/(p^2 + meff^2)^2 * Phi(p^2, t) * exp(-i t/tau_f)
	Note that one has to perform the time integral first to make the p2 integral 
	convergent. This subty has been pointed out in the reference.
	"""           
	
	# Initizalize the solver
	def __init__(self, dt):
		self.dt = dt
		self.lnpT2mD2min = np.log(.01)
		self.lnpT2mD2max = np.log(qmax2/mD2)
		self.lnpT2mD2_grid = np.linspace(self.lnpT2mD2min, self.lnpT2mD2max, 13)
		self.fine_grid = np.linspace(np.log(.001), np.log(2*qmax2/mD2), 201)
		self.results = []
		self.Nt = 1
		self.t = 0.

		self.phi_grid = np.zeros_like(self.lnpT2mD2_grid, dtype=np.complex)
		for i, lnp2 in enumerate(self.lnpT2mD2_grid):
			p2 = mD2*np.exp(lnp2)
			def Kernel(Q2, xfactor):
				lnq2 = np.log(Q2/mD2)
				w1_2 = p2+Q2+mD2*xfactor
				w2_2 = np.sqrt(w1_2**2 - 4*Q2*p2)
				w3_2 = np.abs(p2-Q2)
				
				A = (p2+meff2)*(1 - w3_2/w2_2)
				B = (Q2+meff2)*((p2+meff2)/(Q2+meff2))**2\
					* ((p2+Q2)/2./p2 - w3_2/w2_2 * w1_2/2/p2)
				return (A-B)/(w3_2+m2regulator)
			res = 0.
			for xfactor, color in zip([1, x**2, (1-x)**2], [CA/2., (2*CF-CA)/2., CA/2.]):
				res += color*xfactor\
					  *integrate.quad(Kernel, 0, 8*qmax2/xfactor, args=(xfactor,))[0]
			self.phi_grid[i] = 1j*alphas*T*res/(2*M)
		self.interp = interpolate.interp1d(self.lnpT2mD2_grid, self.phi_grid, kind='cubic', fill_value=0., bounds_error=False)
		self.results.append([self.Phi(ip) for ip in self.fine_grid])
		
	@property
	def x(self):
		return self.lnpT2mD2_grid
	@property
	def y(self):
		return self.phi_grid	
	
	# Phi interpoaltor
	def Phi(self, lnpT2mD2):
		if lnpT2mD2 < self.lnpT2mD2min:
			q2 = mD2*np.exp(lnpT2mD2)
			q02 = mD2*np.exp(self.lnpT2mD2min)
			return self.phi_grid[0] * q2 / q02
		elif lnpT2mD2 > self.lnpT2mD2max:
			q02 = mD2*np.exp(self.lnpT2mD2_grid[-2])
			q12 = mD2*np.exp(self.lnpT2mD2_grid[-1])
			q2 =  mD2*np.exp(lnpT2mD2) 
			l = (q2-q02)/(q12-q02)
			return self.phi_grid[-2]*(1-l) + self.phi_grid[-1]*l
		else:
			return self.interp(lnpT2mD2)

	# C-operator on the wave-function
	def Coperation(self):
		time = self.t - self.dt/2.
		temp_grid = np.zeros_like(self.lnpT2mD2_grid, dtype=np.complex)
		for i, lnp2 in enumerate(self.lnpT2mD2_grid):
			p2 = mD2*np.exp(lnp2)
			p2double = 2*p2
			Phi0 = self.Phi(lnp2)
			def Kernel_Re(Q2, xfactor):
				lnq2 = np.log(Q2/mD2)
				w1_2 = p2+Q2+mD2*xfactor
				w2_2 = np.sqrt(w1_2**2 - 4.*Q2*p2)
				w3_2 = np.abs(p2-Q2)
				
				phase = time*(p2-Q2)/2./M
				A = Phi0.real*(1- w3_2/w2_2 )
				B = (self.Phi(lnq2).real*np.cos(phase) \
					-self.Phi(lnq2).imag*np.sin(phase) )\
				    * ((p2+meff2)/(Q2+meff2))**2\
					* ((p2+Q2) - w3_2/w2_2 * w1_2)/p2double
				return (A-B)/(w3_2+m2regulator)
			def Kernel_Im(Q2, xfactor):
				lnq2 = np.log(Q2/mD2)
				w1_2 = p2+Q2+mD2*xfactor
				w2_2 = np.sqrt(w1_2**2 - 4*Q2*p2)
				w3_2 = np.abs(p2-Q2)
				
				phase = time*(p2-Q2)/2./M
				A = Phi0.imag*(1- w3_2/w2_2 )
				B = (self.Phi(lnq2).real*np.sin(phase) \
					+self.Phi(lnq2).imag*np.cos(phase) )\
				    * ((p2+meff2)/(Q2+meff2))**2\
					* ((p2+Q2) - w3_2/w2_2 * w1_2)/p2double
				return (A-B)/(w3_2+m2regulator)
			re = 0.
			im = 0.
			for xfactor, color in zip([1, x**2, (1-x)**2], [CA/2., (2*CF-CA)/2., CA/2.]):
				re += color*xfactor\
					  *integrate.quad(Kernel_Re, 0, 4*qmax2/xfactor, args=(xfactor,), epsrel=.05)[0]
				im += color*xfactor\
					  *integrate.quad(Kernel_Im, 0, 4*qmax2/xfactor, args=(xfactor,), epsrel=.05)[0]
			temp_grid[i] = alphas*T*(re+1j*im)
		return temp_grid
		
	# Time eovlution
	def evolve(self):
		print(self.t)
		self.t += self.dt
		self.Nt += 1
		self.phi_grid = self.phi_grid - self.Coperation()*self.dt
		self.interp = interpolate.interp1d(self.lnpT2mD2_grid, self.phi_grid, kind='cubic', fill_value=0., bounds_error=False)
		self.results.append([self.Phi(ip) for ip in self.fine_grid])

	# Integrate the wave-function overlap to get the rate 		
	def compute_spectra(self):
		tarray = np.arange(self.Nt)*self.dt	
		self.results = np.array(self.results)
		fr = interpolate.interp2d(tarray, self.fine_grid, self.results.T.real, fill_value=0., kind='cubic')
		fi = interpolate.interp2d(tarray, self.fine_grid, self.results.T.imag, fill_value=0., kind='cubic')
		
		def wave_function_overlap(t, q2):
			deltaE = (q2+meff2)/2/M
			lnq2 = np.log(q2/mD2)
			return q2/(q2+meff2)**2*((fr(t, lnq2) \
				 + 1j*fi(t, lnq2) )*np.exp(-1j*t*deltaE)).real
		
		ts = np.linspace(0.01*tau0, tau0*5, 20)
		q2grid = np.linspace(.01*mD2, qmax2*2, 1000)
		dFdpt2 = np.zeros([len(ts), len(q2grid)])
		for i, t in enumerate(ts):
			for j, q2 in enumerate(q2grid):
				deltaE = (q2+meff2)/2/M
				# make sure dt is small enought for oscillating function
				deltat = 2*np.pi/deltaE/51 
				Nt = int(t / deltat) + 1
				tgrid = np.linspace(0, t, Nt)
				ygrid = wave_function_overlap(tgrid, q2)
				dFdpt2[i,j] = ygrid.sum()*deltat
		
		dFdpt2_interp = interpolate.interp2d(q2grid, ts, dFdpt2, fill_value=0., kind='cubic')
		
		def Rate(t):
			res = integrate.quad(dFdpt2_interp, .001*mD2, 2*qmax2, args=(t,))[0]
			return  res/(2*np.pi)**2*g2*CF*(1+(1-x)**2)/x/E
			
	
		ts = np.linspace(.01*tau0, tau0*5, 40)
		R = np.array([Rate(t) for t in ts])
		return ts, R


if __name__ == "__main__":
	s0 = LO_branching(dt) # make a solver
	q2 = np.linspace(.001*mD2,2*qmax2,1000)
	lnq2 = np.log(q2/mD2)
	# solve the wave-funvtion evolution
	plt.plot(q2/mD2, [s0.Phi(iv).real for iv in lnq2], 'r')
	plt.plot(q2/mD2, [s0.Phi(iv).imag for iv in lnq2], 'b')
	plt.xlabel(r"$p^2/m_D^2$")
	plt.ylabel(r"$\Phi$")
	for i in range(50):
		s0.evolve()
		plt.plot(q2/mD2, [s0.Phi(iv).real for iv in lnq2], 'r')
		plt.plot(q2/mD2, [s0.Phi(iv).imag for iv in lnq2], 'b')
		plt.pause(.1)
	plt.show()
	# integrate the wave function overlap to get the rate
	time, rate = s0.compute_spectra()
	plt.plot(time/5.026, rate, 'r-')
	plt.ylim(0, rate.max()*2)
	plt.xlabel(r"$t$ [fm]")
	plt.ylabel(r"$d\Gamma/d\omega$")
	plt.title(r"$T = {:1.1f}, \omega = {:1.1f}$ in GeV".format(T, omega))
	plt.show()
	
