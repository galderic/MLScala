package org.gp.ml.layers

import org.nd4j.linalg.api.ndarray.INDArray

/**
 * A convenience class that caches the inputs so layers don't have to
 *
 * @param wrapped the layer for which we cache the inputs
 */
case class CachingLayer(wrapped: Layer) {

  var cachedInputs: INDArray = _

  def forward(inputs: INDArray): INDArray = {
    cachedInputs = inputs
    wrapped.forward(inputs)
  }

  def backward(gradient: INDArray): INDArray = {
    wrapped.backward(gradient, cachedInputs)
  }
}
