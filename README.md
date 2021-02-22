This is my own Scala implementation of gradient descent, forward and backward propagation. The purpose is to better 
understand how deep learning works.

I wanted it to be fast enough so it was not only restricted to tiny things. That's why I'm using the [nd4j](https://github.com/deeplearning4j/nd4j) linear algebra libraries for the
matrix multiplications.

The nd4j can make use of the CPU avx2 extensions as well as CUDA for the matrix multiplications. For the
mnist dataset it's not necessary (it's actually faster on cpu than gpu) but it will be surely useful for larger datasets

The provided example automatically downloads the mnist dataset into a temp folder and then trains it with 10 epochs and 
one hidden layer of 100 units. It achieves a 95% accuracy in 50 seconds.

Tons of things missing (convolutional layers, regularization, optimizers, ...) but you first need to get part way there, before 
reaching your destination.