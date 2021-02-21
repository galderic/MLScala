This is my own Scala implementation of gradient descent, forward and backward propagation. The purpose is to better 
understand how deep learning works.

Even though it doesn't pretend to compete with tensorflow or keras, I wanted it to be fast enough
so it was not only restricted to tiny things. That's why I'm using the [nd4j](https://github.com/deeplearning4j/nd4j) linear algebra libraries for the
matrix multiplications.

The nd4j can make use of the CPU avx2 extensions as well as CUDA for the matrix multiplications. For the
mnist dataset it's not necessary (it's actually faster on cpu than gpu) but it will be surely useful for larger datasets

The mnist example downloads when necessary the train and test datasets into a temp folder. 
You can run it with:

`sbt "runMain org.gp.ml.Main"`
 
