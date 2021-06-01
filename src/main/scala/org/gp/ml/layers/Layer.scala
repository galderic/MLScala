package org.gp.ml.layers

import com.typesafe.scalalogging.LazyLogging
import org.nd4j.linalg.api.ndarray.INDArray

trait Layer extends LazyLogging {
  def forward(inputs: INDArray): INDArray

  def backward(gradient: INDArray, inputs: INDArray): INDArray

  def id: String
}
