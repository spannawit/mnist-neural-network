# Neural Network for Digit Recognition (MNIST)

Implementation in python language (with its standard library and Numpy only) of multi-layer neural network with stochastic gradient descent (SGD) optimization method including parameters for learning rate and regularization.

Other functions are grouped in class instead of separated into files for readability.

Following accuracy improvement strategies been done:
1. Normalizing pixel value to range [0,1]
2. Shuffling training data randomly for learning.
3. Mini-batch learning.
4. Weights are initialized from Normal distribution (mean=0, sd=sqrt(#input_unit)) to land around linear region of sigmoid (or tanh) function for fast learning.
   Although ReLu and softmax are available, the network is not optimized for them yet.

* ReLu and softmax does not work well with cross entropy cost function which is used in the code.
  Also usually, they are usually used in the hidden layer and output layer, respectively.

Experiment procedure:
1. Experimented networks are 3-layers NNs (1 hidden layer, 196 activation units) with different activation unit, and cross entropy as cost function instead of a normal error function.
2. Each network will be experimented 5 times with different initialized weights and biases.
3. Network will be learned 50 iterations for each experiment (~1 minute/iteration)
