package org.gp.ml.layers

import com.typesafe.scalalogging.LazyLogging
import org.gp.ml.{Layer, Optimizer, Trainable, WeightsInitializer}
import org.nd4j.linalg.api.ndarray.INDArray

class FullyConnectedLayer(val numInputs: Int, val numOutputs: Int, val optimizer:Optimizer)
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
    val biasGradients = gradient.sum(0)

    optimizer.updateWeights(weights, layerGradients)
    optimizer.updateWeights(bias,biasGradients)

    result
  }
}
