package org.gp.ml.optimizer

import org.nd4j.linalg.api.ndarray.INDArray

trait Optimizer {
  def updateWeights(weights: INDArray, gradient: INDArray)

  def name(): String

  override def toString: String = name()
}
