package org.gp.scratch

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.jcublas.rng.CudaNativeRandom

trait Weights {
  def numInputs: Int

  def numOutputs: Int

  var weights: INDArray = Nd4j.zeros(numInputs, numOutputs)
  xavier(weights)

  var bias = Nd4j.zeros(numOutputs)
  xavier(bias)

  private def xavier(m: INDArray) = {
    Nd4j.rand(m, -1.0d / (Math.sqrt(numInputs)), 1.0d / (Math.sqrt(numInputs)), new CudaNativeRandom())
  }
}
