package org.gp.scratch

import com.typesafe.scalalogging.LazyLogging
import org.nd4j.linalg.api.ndarray.INDArray

trait Layer extends LazyLogging{
  def forwardPass(inputs: INDArray): INDArray
  def backwardPass(gradient: INDArray): INDArray

  def backwardPassWithCheck(gradient: INDArray):INDArray = {
    val result = backwardPass(gradient)
    if (!result.any()) {
      logger.warn("All gradients backpropagated are zero")
    }
    result
  }
}
