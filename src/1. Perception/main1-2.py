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
#Reshape each example into a row, and add a 17th column for the bias term
X=np.hstack((examples.reshape(-1, 16), np.ones((len(y),1))))
#Initialized weights to zeros, this is equivalent to turning each knob to 12 o'clock
w = np.zeros(17)
lr=1.0 #Learning rate

for i in range(1,10):
    yhat=np.dot(X[i%len(y)],w)
    if yhat<=0 and y[i%len(y)]>0:
        print(f"output is {yhat} but we want it to be {y[i%len(y)]}, updating weights.")
        w=w+lr*X[i%len(y)]
    elif yhat>0 and y[i%len(y)]<=0:
        print(f"output is {yhat} but we want it to be {y[i%len(y)]}, updating weights.")
        w=w-lr*X[i%len(y)]
    else:  # ← Now it's part of the if/elif/else chain
        print(f"output is {yhat}, which has the same sign as our target {y[i%len(y)]},"
            f"machine is correct, not updating weights.")

plt.show()