package org.gp.ml

import org.nd4j.linalg.api.ndarray.INDArray

class SquareLossFunction extends LossFunction {
  def loss(y: INDArray, y_pred: INDArray): Double = y.squaredDistance(y_pred)

  def gradient(y: INDArray, y_pred: INDArray): INDArray = y.sub(y_pred).mul(-2)
}
