import numpy as np

# XOR problem - 2 inputs
examples = np.array([[[-1,-1]],
                     [[-1, 1]],
                     [[ 1,-1]],
                     [[ 1, 1]]])

y = np.array([-1, 1, 1, -1])  # XOR: positive when inputs DIFFER

X = np.hstack((examples.reshape(-1, 2), np.ones((len(y), 1))))
w = np.zeros(3)
lr = 1.0

# Track weight history to detect loops
weight_history = []
loop_detected = False
loop_start = -1

for i in range(1, 25):  # Run longer to see the pattern
    yhat = np.dot(X[i % len(y)], w)
    
    # Check if we've seen these weights before (only flag once)
    w_tuple = tuple(w)
    if w_tuple in weight_history and not loop_detected:
        loop_detected = True
        loop_start = weight_history.index(w_tuple)
    weight_history.append(w_tuple)
    
    # Visual separator for each cycle through the 4 examples
    if (i - 1) % 4 == 0:
        print(f"\n{'='*60}")
        cycle_num = (i-1)//4 + 1
        if loop_detected:
            print(f"CYCLE {cycle_num}  🔄 (REPEATING - same as Cycle 1!)")
        else:
            print(f"CYCLE {cycle_num}")
        print(f"{'='*60}")
    
    print(f"\nStep {i}: input={X[i%len(y)][:2]}, target={y[i%len(y)]:+d}, weights={w}")
    
    if yhat <= 0 and y[i % len(y)] > 0:
        print(f"  ŷ={yhat:+.1f} (negative) but want POSITIVE → UPDATE")
        w = w + lr * X[i % len(y)]
        print(f"  New weights: {w}")
    elif yhat > 0 and y[i % len(y)] <= 0:
        print(f"  ŷ={yhat:+.1f} (positive) but want NEGATIVE → UPDATE")
        w = w - lr * X[i % len(y)]
        print(f"  New weights: {w}")
    else:
        print(f"  ŷ={yhat:+.1f} ✓ CORRECT (no update)")

print(f"\n{'='*60}")
print("CONCLUSION: The perceptron never converges.")
print("It cycles through the same 4 weight states forever.")
print("XOR is NOT linearly separable!")
print(f"{'='*60}")