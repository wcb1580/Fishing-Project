# ENGSCI233 Lab: Iteration

# PURPOSE:
# To INVESTIGATE the fisheries management scenario in Task 2

# imports
from functions_itlab import *
from matplotlib import pyplot as plt

# your code here
n = 750000.
f0=130000.
r=0.5
t=50
k=1000000.0
# For quota
t,y=dndt_quota(t, n, r, k, f0)
t1,y1= dndt_kaitiakitanga(50, n, r, k, f0, 0.25)
a,b=dndt_rahui(50, n, r, k, f0, 3)
print(a)
print(b)
f, ax1 = plt.subplots(1, 1)
ax1.set_ylim([0, k])
ax1.set_ylabel('Fish population(1*10^6)')
ax1.plot(t,y, 'b--',label='quota', markersize=6)
ax1.plot(t,y1, 'go', label='kaitiakitanga', zorder=2)
ax1.plot(t,b,'r-',label='rahui',zorder=2)
ax1.set_xlabel('time(years)')
ax1.legend(loc=3)
plt.show()
save_figure = True
if not save_figure:
        plt.show()
else:
        plt.savefig('task2.png', dpi=300)

