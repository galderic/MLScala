package org.gp.ml

import org.nd4j.linalg.api.ndarray.INDArray

case class Batch(features: INDArray, labels: INDArray, index: Int) {
  def numSamples(): Long = {
    labels.rows()
  }
}
