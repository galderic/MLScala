package org.gp.ml.loss

import org.nd4j.linalg.api.ndarray.INDArray

trait LossFunction {
  def loss(labels: INDArray, predictions: INDArray): Double

  def gradient(labels: INDArray, predictions: INDArray): INDArray
}
