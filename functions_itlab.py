# ENGSCI233: Lab - Iteration and Stability

# PURPOSE:
# To IMPLEMENT explicit Runge-Kutta ODE solution methods.

# imports
import numpy as np


def step_ieuler(f, tk, yk, h, args=None):
	'''The function step_ieuler() perform a single step of improved euler method

		Parameters
		----------
		f: callable derivate function
		tk : float
			Initial value of independent variable.
		h : float
			step size
		yk: float
			initial value of solution
		args: iterable
			Optional parameters to pass into derivative function.
		Returns
		-------
		yk1: float
			solution of the improved euler step
		Example
		-------
		>> def dydt1(t,y,a,b):
    		return a*t-b*y
    	>> yk = step_ieuler(dydt1, 0.0, 1.0, 2.0, args=[3.,2.])
    	>> yk = 11.0




	'''
	if args is None:
		args = []
	f0=f(tk,yk,*args)
	f1=f(tk+h,yk+h*f0,*args)
	yk1=yk+h*((f0+f1)/2) # perform improved euler method for a step
	return yk1

def step_rk4(f, tk, yk, h, args=None):
	'''The function step_rk4() perform a single step of RK4 method

			Parameters
			----------
			f: callable derivate function
			tk : float
				Initial value of independent variable.
			h : float
				step size
			yk: float
				initial value of solution
			args: iterable
				Optional parameters to pass into derivative function.
			Returns
			-------
			yk1: float
				solution of the RK4 step
			Example
			-------
			>> def dydt(t,y):
    			return t-y
    		>> yk = step_rk4(dydt, 0.0, 1.0, 2.0, args=None)
    		>> yk = 5/3

		'''
	if args is None:
		args = []
	f0=f(tk,yk,*args)
	f1=f(tk+h/2,yk+(h*f0)/2,*args)
	f2=f(tk+h/2,yk+(h*f1)/2,*args)
	f3 = f(tk + h , yk + (h * f2),*args)
	yk1=yk+(h*(f0+2*f1+2*f2+f3)/6) # perform single rk4 method with a step
	return yk1




def solve_explicit_rk(f, t0, t1, y0, h, method='rk4', args=None):
	"""	Compute solution of initial value ODE problem using explicit RK method.

		Parameters
		----------
		f : callable
			Derivative function.
		t0 : float
			Initial value of independent variable.
		t1 : float
			Final value of independent variable.
		y0 : float
			Initial value of solution.
		h : float
			Step size.
		method : str
			String specifying RK method, either 'rk4' or 'euler'. Default is 'rk4'.
		args : iterable
			Optional parameters to pass into derivative function.

		Returns
		-------
		t : array-like
			Independent variable at solution.
		y : array-like
			Solution.

		Notes
		-----
		Assumes that order of inputs to f is f(t, y, *args).
	"""
	length=(t1-t0)//h+1
	if (t1-t0)/h != (t1-t0)//h:
		length+=1 # Set up the length of the return array
	t = np.zeros(int(length))
	ysolution = np.zeros(int(length))
	i=0
	step = t1 // h
	if args is None:
		args = []
	if method=='rk4': # perfrom RK4 method for steps
		T=t0;
		y=y0;
		while (T <= step*h): # calculate the y value for the first one to the one before the last one
			yk1=step_rk4(f, T, y, h, args)
			t[i]=T #record related t value
			ysolution[i]=y
			T+=h
			y=yk1
			i+=1
		if (t1-t0)/h != (t1-t0)//h: # perform the last step so that it ensure that T = t1 no matter if t1 is dividable by h
			if T > t1:
				T =t1
			result = step_rk4(f, T, y, t1-T, args)
			t[i] = T
			ysolution[i] = result
	if method=='ieuler': # Perform improved euler method for steps
		T=t0;
		y=y0
		while T <= step*h : # calculate the y value for the first one to the one before the last one
			yk1=step_ieuler(f, T, y, h, args)
			t[i]=(T) #record related t value
			ysolution[i]=y
			T+=h
			y=yk1
			i+=1
		if (t1 - t0) / h != (t1 - t0) // h: # perform the last step so that it ensure that T = t1 no matter if t1 is dividable by h
			if T > t1:
				T = t1
			result = step_ieuler(f, T, y, t1 - T, args)
			t[i] = T
			ysolution[i] = result
	return t,ysolution

def dndt(x,y,n,r,k,f0):
	'''The function dndt() is a function indicates the rate of change of fish population when fishing is permitted

				Parameters
				----------
				x:  float
					x input
				y: 	float
					y input
				n: 	float
					fish population
				r : float
					birth rate
				k: float
					constant for population capacity
				f0: float
					amount of fish caught.
				Returns
				-------
				dndt: float
					the change in fish population
			'''

	dndt = r * n * (1 - n / k) - f0
	return dndt
def dndtrest(x,y,n,r,k):
	'''The function dndt() is a function indicates the rate of change of fish population when fishing is not permitted

					Parameters
					----------
					x:  float
						x input
					y: 	float
						y input
					n: 	float
						fish population
					r : float
						birth rate
					k: float
						constant for population capacity
					Returns
					-------
					dndt: float
						the change in fish population
				'''
	dndt = r * n * (1 - n / k)
	return dndt






def dndt_quota(t, n, r, k, f0):
	'''The function dndt() is a function represent the change in fish population using quota method for fishing using rk4 method

		Parameters
		----------
		t:  float
			years
		n: 	float
			fish population
		r : float
			birth rate
		k: float
			constant for population capacity
		Returns
		-------
		t: an array
			years in record
		y: an array
			fish population each year
		'''
	initial = 0
	y = np.zeros(t + 1)
	year = np.zeros(t + 1) # set up the return array
	i = 0
	while initial <= t:
		y[i] = n #record population
		new = step_rk4(dndt, t, n, 1.0, args=[n, r, k, f0]) # f0 is constant always
		n = new
		year[i]=initial # record years
		initial+=1
		i+=1

	return year,y









def dndt_kaitiakitanga(t, n, r, k, f0, fr):
	'''The function dndt() is a function represent the change in fish population using kaitiakitanga method for fishing using rk4 method

			Parameters
			----------
			t:  float
				years
			n: 	float
				fish population
			r : float
				birth rate
			k: float
				constant for population capacity
			fr: float
				a percentage that the amount of fishing this year can not excess respective to the whole fish population
			Returns
			-------
			t: an array
				years in record
			y: an array
				fish population each year
			'''
	initial = 0
	y = np.zeros(t + 1)
	year = np.zeros(t + 1)
	i = 0
	while initial <= t:
		if n*fr>=f0: # if fixed fishing number is greater than a fixed fraction of the population
			y[i] = n #record population
			new = step_rk4(dndt, t, n, 1.0, args=[n, r, k, f0])
			n = new
			year[i] = initial #record tear
			initial += 1
			i += 1
		else: #  if the fishing number does not exceed a fixed fraction of the population
			y[i] = n #record population
			new = step_rk4(dndt, t, n, 1.0, args=[n, r, k, n*fr])
			n = new
			year[i] = initial #record year
			initial += 1
			i += 1
	return year,y


def dndt_rahui(t, n, r, k, f0, x):
	'''The function dndt() is a function represent the change in fish population using rahui method for fishing using rk4 method

				Parameters
				----------
				t:  float
					years
				n: 	float
					fish population
				r : float
					birth rate
				k: float
					constant for population capacity
				x: float
					years when fishing is not permitted
				Returns
				-------
				t: an array
					years in record
				y: an array
					fish population each year
				'''
	swap = t//x
	y = np.zeros(t + 1)
	year = np.zeros(t+1) #set up the return array
	for initial in range(swap): #record in years for the range till the maximum year can be fully divided by x
		if (initial%2!=0 and t-1-initial*x>=x-1): # for period that fishing is not perimitted
			for i in range(x):
				y[initial*x+i] = n
				new = step_rk4(dndtrest, t, n, 1.0, args=[n, r, k])
				n = new
		if (initial%2==0 and t-1-initial*x>=x-1): # for period that fishing is allowed
			for i in range(x):
				y[initial*x+i] = n
				new = step_rk4(dndt, t, n, 1.0, args=[n, r, k,f0])
				n = new
	result = np.where(y == 0)
	if (y[result[0][0]-1]>=y[result[0][0]-2]): # perform calculation for the last few years between maximum year that is fully divided by x and the total year t.
		while result[0][0]<t+1:
			y[result[0][0]] = n
			new = step_rk4(dndt, t, n, 1.0, args=[n, r, k, f0])# if perivous period is in resst, then the last few years will enable fishing
			n = new
			result[0][0]+=1
	if (y[result[0][0]-1]<y[result[0][0]-2]):
		while result[0][0]<t+1:
			y[result[0][0]] = n
			new = step_rk4(dndtrest, t, n, 1.0, args=[n, r, k])# if perivous period is not in rest, then the last few years will not enable fishing
			n = new
			result[0][0]+=1
	for i in range(t+1):
		year[i]=i
	return result,y





















	return year,y

























