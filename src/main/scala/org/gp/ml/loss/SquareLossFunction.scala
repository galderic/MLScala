package org.gp.ml.loss

import org.nd4j.linalg.api.ndarray.INDArray

case class SquareLossFunction() extends LossFunction {
  def loss(labels: INDArray, predictions: INDArray): Double = labels.squaredDistance(predictions)

  def gradient(labels: INDArray, predictions: INDArray): INDArray = labels.sub(predictions).mul(-2)
}
