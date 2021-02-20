package org.gp.ml

import org.nd4j.linalg.api.ndarray.INDArray

trait Trainable {
  protected def getWeights: INDArray

  def summary(): String = {
    s"Weights mean and stdev for ${getWeights.rows()}x${getWeights.columns()} : ${getWeights.meanNumber()}, ${getWeights.stdNumber()}"
  }
}
