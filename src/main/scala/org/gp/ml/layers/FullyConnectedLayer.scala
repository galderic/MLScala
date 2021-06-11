package org.gp.ml.layers

import com.typesafe.scalalogging.LazyLogging
import org.gp.ml.optimizer.Optimizer
import org.gp.ml.weights.WeightsInitializer
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

class FullyConnectedLayer(val numInputs: Int, val numOutputs: Int, val optimizer: Optimizer, val id: String = "fullyConnectedLayer")
  extends Layer with Trainable with LazyLogging {

  val weights: INDArray = WeightsInitializer.xavier(numInputs, numOutputs)
  val layerGradients: INDArray = WeightsInitializer.xavier(numInputs, numOutputs)

  var bias: INDArray = WeightsInitializer.fromValue(numOutputs, 0.01d)
  var biasGradients: INDArray = WeightsInitializer.fromValue(numOutputs, 0.01d)

  var cachedResult: Option[INDArray] = None

  override def forward(inputs: INDArray): INDArray = {
    Nd4j.gemm(inputs,weights,false,false).add(bias)

    //inputs.mmul(weights).add(bias)
  }

  override def backward(gradient: INDArray, lastInputs: INDArray): INDArray = {

    // gradient w.r. the layer inputs
    //val result = gradient.mmul(weights.transpose())

    if (cachedResult.isDefined) {
      Nd4j.gemm(gradient, weights, cachedResult.get, false, true, 1.0, 0.0)
    } else {
      val newResult = Nd4j.gemm(gradient, weights, false, true)
      cachedResult = Some(newResult)
    }
    // gradient w.r. the weights
    //val layerGradients = lastInputs.transpose().mmul(gradient)
    //val layerGradients =
    Nd4j.gemm(lastInputs, gradient, layerGradients, true, false, 1.0, 0.0)
    gradient.sum(biasGradients, 0)

    optimizer.updateWeights(weights, layerGradients)
    optimizer.updateWeights(bias, biasGradients)

    cachedResult.get
  }
}
