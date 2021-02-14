package org.gp.scratch

import com.typesafe.scalalogging.LazyLogging
import org.nd4j.linalg.api.ndarray.INDArray

class FullyConnectedLayer(val numInputs: Int, val numOutputs: Int) extends TrainableLayer with LazyLogging {

  var lastInputs: INDArray = _

  override def forwardPass(inputs: INDArray): INDArray = {
    lastInputs = inputs.dup()
    inputs.mmul(weights).add(bias)
  }

  override def backwardPass(gradient: INDArray): INDArray = {
    val weightsBeforeUpdate = weights.dup
    val biasBeforeUpdate = bias.dup()

    val layerGradients = lastInputs.transpose().mmul(gradient)

    val weightDiff = layerGradients.mul(.01)

    logger.info(s"weightDiff mean:${weightDiff.mean(1).mean(0)}")
    logger.info(s"weightDiff max:${weightDiff.amax(1).amax(0)}")

    val biasDiff = gradient.sum(0)

    if (!weightDiff.any()) {
      logger.warn("Weights are not changing")
    }

    weights = weightsBeforeUpdate.sub(weightDiff)
    bias = biasBeforeUpdate.sub(biasDiff)

    gradient.mmul(weightsBeforeUpdate.transpose())
  }
}
