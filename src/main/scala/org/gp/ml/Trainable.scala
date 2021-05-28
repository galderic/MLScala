package org.gp.ml

import org.nd4j.linalg.api.ndarray.INDArray

trait Trainable extends Layer{
  val weights: INDArray
  var bias: INDArray

  def summary(): String = {
    s"shape: [${weights.shape().mkString(",")}] mean:${weights.meanNumber()} std:${weights.stdNumber()}"
  }
}
