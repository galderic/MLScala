package org.gp.scratch

import org.nd4j.linalg.api.ndarray.INDArray

trait HasWeights {
  def createWeights(inputs: Int): INDArray
}
