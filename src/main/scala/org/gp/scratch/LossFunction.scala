package org.gp.scratch

import org.nd4j.linalg.api.ndarray.INDArray

trait LossFunction {
  def cost(y: INDArray, y_pred: INDArray): Double

  def derivative(y: INDArray, y_pred: INDArray): INDArray
}
