package org.gp.ml.layers

import org.gp.ml.optimizer.Optimizer
import org.nd4j.linalg.api.ndarray.INDArray

trait Trainable extends Layer {
  def weights: INDArray

  def bias: INDArray

  def optimizer: Optimizer

  def summary(): String = {
    s"shape: [${weights.shape().mkString(",")}] mean:${weights.meanNumber()} std:${weights.stdNumber()} optimizer:$optimizer"
  }
}
