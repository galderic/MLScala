package org.gp.scratch

import org.nd4j.linalg.api.ndarray.INDArray

class SquareLossFunction extends LossFunction {
  def cost(y: INDArray, y_pred: INDArray) = y.squaredDistance(y_pred)
  def derivative(y: INDArray, y_pred: INDArray): INDArray = {
    y.sub(y_pred).mul(-1).div(2)
  }
}
