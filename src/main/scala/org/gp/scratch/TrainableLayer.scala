package org.gp.scratch

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

trait TrainableLayer extends Layer {
  def numInputs: Int
  def numOutputs: Int

  val weights: INDArray = Nd4j.rand(numInputs, numOutputs)
}
