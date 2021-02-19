package org.gp.scratch

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.jcublas.rng.CudaNativeRandom

object WeightsInitializer {
  def toValue(numOutputs: Int, value:Double): INDArray = {
    Nd4j.zeros(numOutputs).assign(value)
  }

  def xavier(numInputs: Int, numOutputs: Int): INDArray = {
    var result = Nd4j.zeros(numInputs, numOutputs)
    Nd4j.rand(result, -1.0d / Math.sqrt(numInputs), 1.0d / Math.sqrt(numInputs), new CudaNativeRandom())
    result
  }
}
