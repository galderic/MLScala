package org.gp.scratch

import org.nd4j.linalg.api.ndarray.INDArray

trait Layer {
  def forwardPass(inputs: INDArray): INDArray
  def backwardPass(gradient: INDArray): INDArray
}
