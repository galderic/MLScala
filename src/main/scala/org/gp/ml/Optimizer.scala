package org.gp.ml

import org.nd4j.linalg.api.ndarray.INDArray

trait Optimizer {
  def updateWeights(weights: INDArray, gradient: INDArray)
}
