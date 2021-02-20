package org.gp.ml

import org.nd4j.linalg.api.ndarray.INDArray

case class Batch(features: INDArray, labels: INDArray) {
  def numSamples(): Long = {
    labels.rows()
  }
}
