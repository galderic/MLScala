package org.gp.scratch

import org.nd4j.linalg.api.ndarray.INDArray

trait Trainable {
  def getWeights: INDArray
}
