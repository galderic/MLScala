package org.gp.ml

import com.typesafe.scalalogging.LazyLogging
import org.nd4j.linalg.api.ndarray.INDArray

/**
 * A layer keeps a cache of the last inputs forwarded, so the can be used in the backpropagation to calculate the
 * gradient. It also checks if the backpropagated gradients have vanished.
 */
trait Layer extends LazyLogging {
  protected def forward(inputs: INDArray): INDArray

  protected def backward(gradient: INDArray): INDArray

  private var cachedInputs: INDArray = _

  def forwardPass(input: INDArray): INDArray = {
    cachedInputs = input
    forward(input)
  }

  def backwardPass(gradient: INDArray): INDArray = {
    val result = backward(gradient)
    if (!result.any()) {
      logger.warn("All gradients backpropagated are zero")
    }
    result
  }

  def lastInputs(): INDArray = cachedInputs
}
