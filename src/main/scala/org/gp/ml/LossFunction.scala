package org.gp.ml

import org.nd4j.linalg.api.ndarray.INDArray

trait LossFunction {
  def loss(y: INDArray, y_pred: INDArray): Double

  def gradient(y: INDArray, y_pred: INDArray): INDArray
}
