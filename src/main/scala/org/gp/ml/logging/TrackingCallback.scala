package org.gp.ml.logging

import org.gp.ml.layers.Layer
import org.nd4j.linalg.api.ndarray.INDArray

trait TrackingCallback {
  def afterForward(layer: Layer, inputs: INDArray, outputs: INDArray, batchNum: Int)

  def afterBackward(layer: Layer, cachedInputs: INDArray, inputGradient: INDArray, outputGradient: INDArray, batchNum: Int)

  def labelsUsedInCostFunction(labelVector: INDArray)
}
