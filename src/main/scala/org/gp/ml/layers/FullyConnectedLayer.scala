package org.gp.ml.layers

import com.typesafe.scalalogging.LazyLogging
import org.gp.ml.optimizer.Optimizer
import org.gp.ml.weights.WeightsInitializer
import org.nd4j.linalg.api.ndarray.INDArray

case class FullyConnectedLayer(numInputs: Int, numOutputs: Int, optimizer: Optimizer, id: String = "fullyConnectedLayer")
  extends Layer with Trainable with LazyLogging {

  val weights: INDArray = WeightsInitializer.xavier(numInputs, numOutputs)
  var bias: INDArray = WeightsInitializer.bias(numOutputs, 0.01d)

  override def forward(inputs: INDArray): INDArray = {
    inputs.mmul(weights).add(bias)
  }

  override def backward(gradient: INDArray, lastInputs: INDArray): INDArray = {

    // gradient w.r. the layer inputs
    val result = gradient.mmul(weights.transpose())

    // gradient w.r. the weights
    val layerGradients = lastInputs.transpose().mmul(gradient)
    val biasGradients = gradient.sum(0)

    optimizer.updateWeights(weights, layerGradients)
    optimizer.updateWeights(bias, biasGradients)

    result
  }
}
