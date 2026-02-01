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

# Reshape each example into a row, and add a 17th column for the bias term
# Bias term is a like a switch that is "always on", an extra parameter that doesn't depend the input
X=np.hstack((examples.reshape(-1, 16), np.ones((len(y),1))))

#Initialized weights to zeros, this is equivalent to turning each knob to 12 o'clock
w = np.zeros(17)
lr=1.0 #Learning rate

i=1 #Start with index 1, converges a little faster than starting at index 0
yhat=np.dot(X[i],w) #Compute perceptron output by taking dot product of example X and weights.
yhat, y[i] #machine outputs 0, but we want it to output +1

# Update weight following perceptron learning rule
# adding our learning rate times our example is equivalent
# to turning up all our dials that are switched on, and turning
# down all our dails that are switched off
w=w+lr*X[i]

#Adding our learning rate times our example now makes our weights look like our first example.
w[:16].reshape(4,4)

i+=1 #Increment our counter i

yhat=np.dot(X[i],w) #Compute perceptron output

w=w-lr*X[i] #Machine output a +, but we wanted -, so subract learning rate time examples

i+=1 #Increment our counter i

yhat=np.dot(X[i],w) #Compute perceptron output
yhat, y[i] #machine outputs +, but we want it to output -

w=w-lr*X[i] #Machine output a +, but we wanted -, so subract learning rate time examples

w[:16].reshape(4,4)

i=0 #We've reached the end our our examples (index 3, so start over)
yhat=np.dot(X[i],w) #Compute perceptron output


#print(yhat, y[i], w)

#Cycle back through examples, print machine output and target output for each
for i in range(4):
    yhat=np.dot(X[i],w) #Compute perceptron output
    print(yhat, y[i])