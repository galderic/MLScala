package org.gp.scratch

import org.nd4j.linalg.api.ndarray.INDArray

trait Layer {
  def numInputs: Int
  def numOutputs: Int
  def forwardPass(inputs: INDArray): INDArray
}
