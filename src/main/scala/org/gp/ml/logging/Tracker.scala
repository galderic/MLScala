package org.gp.ml.logging

import org.gp.ml.layers.Layer
import org.nd4j.linalg.api.ndarray.INDArray

trait Tracker {
  def afterForward(layer: Layer, inputs: INDArray, outputs: INDArray, batchNum: Int): Unit

  def afterBackward(layer: Layer, cachedInputs: INDArray, inputGradient: INDArray, outputGradient: INDArray, batchNum: Int): Unit

  def labelsUsedInCostFunction(labelVector: INDArray): Unit

  def lossFunctionGradient(gradients: INDArray): Unit
}
