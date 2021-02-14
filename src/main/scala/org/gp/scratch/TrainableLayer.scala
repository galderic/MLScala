package org.gp.scratch

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.rng.{DefaultRandom, Random}
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.jcublas.rng.CudaNativeRandom
import org.nd4j.rng.NativeRandom

trait TrainableLayer extends Layer {
  def numInputs: Int
  def numOutputs: Int

  var weights: INDArray = Nd4j.rand(numInputs, numOutputs)
  //public static INDArray rand(INDArray target,  double min, double max, @NonNull org.nd4j.linalg.api.rng.Random rng) {
  Nd4j.rand(weights, -1.0d / (Math.sqrt(numInputs)), 1.0d / (Math.sqrt(numInputs)), new CudaNativeRandom())
  var bias = Nd4j.rand(numOutputs)
  Nd4j.rand(bias, -1.0d / (Math.sqrt(numInputs)), 1.0d / (Math.sqrt(numInputs)), new CudaNativeRandom())
}
