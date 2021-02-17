package org.gp.scratch

import com.typesafe.scalalogging.LazyLogging
import org.nd4j.linalg.api.ndarray.INDArray

trait Layer extends LazyLogging {
  protected def forwardPass(inputs: INDArray): INDArray

  protected def backwardPass(gradient: INDArray): INDArray

  private var cachedInputs: INDArray = _

  def forwardPassWithCheck(input: INDArray): INDArray = {
    cachedInputs = input
    forwardPass(input)
  }

  def backwardPassWithCheck(gradient: INDArray): INDArray = {
    val result = backwardPass(gradient)
    if (!result.any()) {
      logger.warn("All gradients backpropagated are zero")
    }
    result
  }

  def lastInputs():INDArray = cachedInputs
}
