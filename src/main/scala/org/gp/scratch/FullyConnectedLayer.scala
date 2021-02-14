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

    val learningRate=.03
    //val weightDiff = layerGradients.mul(1.2)
    val weightDiff = layerGradients.mul(learningRate)

//    logger.info(s"weightDiff ${numInputs}x${numOutputs} mean:${weightDiff.mean(1).mean(0)}")
//    logger.info(s"weightDiff ${numInputs}x${numOutputs} max:${weightDiff.amax(1).amax(0)}")

    val biasDiff = gradient.sum(0).mul(learningRate)

//    logger.info(s"biasDiff ${numInputs}x${numOutputs} mean:${biasDiff.mean(0)}")
//    logger.info(s"biasDiff ${numInputs}x${numOutputs} max:${biasDiff.amax(0)}")


    if (!weightDiff.any()) {
      logger.warn("Weights are not changing")
    }

    weights = weightsBeforeUpdate.sub(weightDiff)
    bias = biasBeforeUpdate.sub(biasDiff)

    gradient.mmul(weightsBeforeUpdate.transpose())
  }
}
