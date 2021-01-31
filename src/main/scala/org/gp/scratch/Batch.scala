package org.gp.scratch

import org.nd4j.linalg.api.ndarray.INDArray

case class Batch(val features:INDArray, val labels:INDArray) {
  def size():Long = {
    labels.size(0)
  }
}
