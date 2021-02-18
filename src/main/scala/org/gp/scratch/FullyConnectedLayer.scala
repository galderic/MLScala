package org.gp.scratch

import com.typesafe.scalalogging.LazyLogging
import org.nd4j.linalg.api.ndarray.INDArray

class FullyConnectedLayer(val numInputs: Int, val numOutputs: Int, val learningRate: Double)
  extends Layer with Trainable with LazyLogging {

  var weights: INDArray = WeightsInitializer.xavier(numInputs, numOutputs)

  var bias = WeightsInitializer.zeros(numOutputs)

  override def forward(inputs: INDArray): INDArray = {
    inputs.mmul(weights).add(bias)
  }

  override def backward(gradient: INDArray): INDArray = {
    val weightsBeforeUpdate = weights.dup
    val biasBeforeUpdate = bias.dup()

    val layerGradients = lastInputs.transpose().mmul(gradient)

    val weightDiff = layerGradients.mul(learningRate)

    val biasDiff = gradient.sum(0).mul(learningRate)

    if (!weightDiff.any()) {
      logger.warn("Weights are not changing")
    }

    weights = weightsBeforeUpdate.sub(weightDiff)
    bias = biasBeforeUpdate.sub(biasDiff)

    gradient.mmul(weightsBeforeUpdate.transpose())
  }

  override def getWeights: INDArray = weights
}
