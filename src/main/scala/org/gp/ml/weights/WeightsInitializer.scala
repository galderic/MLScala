package org.gp.ml.weights

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.cpu.nativecpu.rng.CpuNativeRandom
import org.nd4j.linalg.factory.Nd4j

object WeightsInitializer {
  def bias(numOutputs: Int, value: Double): INDArray = {
    Nd4j.zeros(numOutputs).assign(value)
  }

  def xavier(numInputs: Int, numOutputs: Int): INDArray = {
    val random = new CpuNativeRandom(543543L)

    var result = Nd4j.rand(numInputs, numOutputs, random)
    Nd4j.rand(result, -1.0d / Math.sqrt(numInputs), 1.0d / Math.sqrt(numInputs), random)
    result
  }
}
