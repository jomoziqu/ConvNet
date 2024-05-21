from ad import dual

def relu(x):
    # ReLU activation function
    return dual(max(x.val, 0), x.grad if x.val > 0 else 0)

def convnet(x_values):
    # Convert input numerical values to dual objects
    x = [dual(val, 0) for val in x_values]
    
    # Define the weight parameters
    w1 = dual(1.2, 1)  # Gradient of w1 is 1
    w2 = dual(-0.2, 0) 
    v = [dual(-0.3, 0), dual(0.6, 0), dual(1.3, 0), dual(-1.5, 0)]
    
    # Compute the hidden layer values with ReLU activation
    z1 = relu(x[0] * w1 + x[1] * w2)
    z2 = relu(x[2] * w1 + x[3] * w2)
    z3 = relu(x[4] * w1 + x[0] * w2)
    z4 = relu(x[1] * w1 + x[2] * w2)
    z = [z1, z2, z3, z4]
    
    # Compute the output value y
    y = v[0] * z1 + v[1] * z2 + v[2] * z3 + v[3] * z4
    
    return y, z

# Testing the convnet function with user input
x_values = [1, 2, 3, 4, 5]
y, z = convnet(x_values)
print(f"\nOutput value y: {y}, \n\nHidden layer values z: {z}\n\n")



