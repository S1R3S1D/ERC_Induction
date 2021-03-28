import numpy as np
import matplotlib.pyplot as plt

np.random.seed(101)

#inputs and outputs
inputs=np.array([[0,0],[0,1],[1,0],[1,1]])
outputs=np.array([[0],[1],[1],[0]])

#number of inputs, hidden layer neurons, outputs
n_inputs,n_layer,n_outputs=2,4,1

#initializing weights and biases
layer_weights=np.random.random((n_inputs,n_layer))*2-1
layer_bias=np.random.random((1,4))*2-1
output_weights=np.random.random((n_layer,n_outputs))
output_bias=np.random.random((n_outputs))

#sigmoid and its derivative
def sigmoid(x):
    return 1/(1+np.exp(-x))
def ddx_sigmoid(x):
    return x*(1-x)
    
#epochs and learning rate
epochs=50000
lr=0.3

#Neural Network implementation
cfl=[]
for i in range(epochs):

    #forward-prop
    layer_outputs=sigmoid(np.dot(inputs,layer_weights)+layer_bias)
    predicted_output=sigmoid(np.dot(layer_outputs,output_weights)+output_bias)

    #backprop
    error=outputs-predicted_output
    d_predicted_output=ddx_sigmoid(predicted_output)*error
    error_layer=d_predicted_output.dot(output_weights.T)
    d_layer_output=error_layer*ddx_sigmoid(layer_outputs)

    #updating Weights and biases
    output_weights+=d_predicted_output.T.dot(layer_outputs).T*lr
    output_bias+=np.sum(d_layer_output)*lr
    layer_weights+=(np.dot(d_layer_output,inputs).T)*lr
    layer_bias+=(np.sum(d_layer_output,axis=0))*lr


    #plotting cost function wrt number of iterations
    costf=np.sum(((outputs-predicted_output)**2)/2)
    cfl.append(costf)


#results of the trained network
print(predicted_output)
plt.plot([x for x in range(epochs)],cfl)
plt.show()
