import numpy as np
import matplotlib.pyplot as plt

#Manually Code up examples
examples=np.array([[[1,1,1,-1], #T Shifted Left
[-1,1,-1,-1],
[-1,1,-1,-1],
[-1,1,-1,-1]],
[[-1,1,1,1], #T Shifted Right
[-1,-1,1,-1],
[-1,-1,1,-1],
[-1,-1,1,-1]],
[[-1,-1,1,-1], #J Shifted Left
[-1,-1,1,-1],
[1,-1,1,-1],
[1,1,1,-1]],
[[-1,-1,-1,1], #J Shifted Right
[-1,-1,-1,1],
[-1,1,-1,1],
[-1,1,1,1]]])

fig=plt.figure(0,(6,6))
for i in range(len(examples)):
    fig.add_subplot(2,2,i+1)
    plt.imshow(examples[i])

# Setup labels - we want our machine to output positive voltage to T shapes,
# our first 2 examples are Ts - so we'll set these values to +1, our second to
# examples are Js - so we'll set these to -1s
y=np.array([1,1,-1,-1])


plt.show()