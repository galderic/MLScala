package org.gp.scratch

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.weightinit.WeightInit

trait TrainableLayer extends Layer {
  def numInputs: Int
  def numOutputs: Int

  var weights: INDArray = Nd4j.rand(numInputs, numOutputs)
  var bias = Nd4j.rand(numOutputs)

}
