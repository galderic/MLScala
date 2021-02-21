package org.gp.ml.layers

import com.typesafe.scalalogging.LazyLogging
import org.gp.ml.{Layer, Trainable, WeightsInitializer}
import org.nd4j.linalg.api.ndarray.INDArray

class FullyConnectedLayer(val numInputs: Int, val numOutputs: Int, val learningRate: Double)
  extends Layer with Trainable with LazyLogging {

  override def getWeights: INDArray = weights

  var weights: INDArray = WeightsInitializer.xavier(numInputs, numOutputs)

  var bias:INDArray = WeightsInitializer.fromValue(numOutputs, 0.01d)

  override def forward(inputs: INDArray): INDArray = {
    inputs.mmul(weights).add(bias)
  }

  override def backward(gradient: INDArray): INDArray = {

    val result = gradient.mmul(weights.transpose())

    val layerGradients = lastInputs().transpose().mmul(gradient)

    val weightDiff = layerGradients.mul(learningRate)

    val biasDiff = gradient.sum(0).mul(learningRate)

    if (!weightDiff.any()) {
      logger.warn("Weights are all zero")
    }

    weights.subi(weightDiff)
    bias.subi(biasDiff)

    result
  }
}
