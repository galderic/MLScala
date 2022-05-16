package org.gp.ml.logging

import org.gp.ml.layers.Layer
import org.nd4j.linalg.api.ndarray.INDArray

case class NoOpTracker() extends Tracker {
  override def afterForward(layer: Layer, inputs: INDArray, outputs: INDArray, batchNum: Int): Unit = {}

  override def afterBackward(layer: Layer, cachedInputs: INDArray, inputGradient: INDArray, outputGradient: INDArray, batchNum: Int): Unit = {}

  override def labelsUsedInCostFunction(labelVector: INDArray): Unit = {}

  override def lossFunctionGradient(gradients: INDArray): Unit = {}
}
