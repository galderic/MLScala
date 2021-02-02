package org.gp.scratch

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

class FullyConnectedLayer(val numInputs: Int, val numOutputs: Int) extends TrainableLayer {
  override def forwardPass(inputs: INDArray): INDArray =  {
    println(s"inputs shape:${inputs.shape().mkString(",")} weights shape:${weights.shape().mkString(",")}")
    inputs.transpose().mmul(weights)
  }
}
