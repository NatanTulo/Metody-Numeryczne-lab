import numpy as np
import matplotlib.pyplot as plt

x=np.linspace(0,4*np.pi,100)
y=np.sin(x)
a=[[1,2],[2,4]]
print(np.array(a))
plt.plot(x,y,"r.")
plt.show()