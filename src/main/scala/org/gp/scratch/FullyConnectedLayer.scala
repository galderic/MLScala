package org.gp.scratch

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

class FullyConnectedLayer(val numInputs: Int, val numOutputs: Int) extends TrainableLayer {

  var lastInputs:INDArray = _

  override def forwardPass(inputs: INDArray): INDArray =  {
//    println(s"inputs shape:${inputs.shape().mkString(",")} weights shape:${weights.shape().mkString(",")}")
//    println(s"weights:${weights}")
    lastInputs = inputs.dup()
    inputs.transpose().mmul(weights).add(bias)
  }

  override def backwardPass(gradient: INDArray): INDArray = {
    val weightsBeforeUpdate = weights.dup

    val layerGradients = lastInputs.mmul(gradient.transpose())

    weights = weightsBeforeUpdate.sub(layerGradients.mul(.1))

    gradient.transpose().mmul(weightsBeforeUpdate.transpose())
  }
}
