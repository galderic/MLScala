package org.gp.scratch

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.jcublas.rng.CudaNativeRandom

trait TrainableLayer extends Layer {
  def numInputs: Int

  def numOutputs: Int

  var weights: INDArray = Nd4j.rand(numInputs, numOutputs)

  Nd4j.rand(weights, -1.0d / (Math.sqrt(numInputs)), 1.0d / (Math.sqrt(numInputs)), new CudaNativeRandom())
  var bias = Nd4j.rand(numOutputs)
  Nd4j.rand(bias, -1.0d / (Math.sqrt(numInputs)), 1.0d / (Math.sqrt(numInputs)), new CudaNativeRandom())
}
