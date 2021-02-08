package org.gp.scratch

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.ops.transforms.Transforms

class Softmax() extends Layer {
  override def forwardPass(inputs: INDArray): INDArray = {
    Transforms.softmax(inputs)
  }

  override def backwardPass(gradient: INDArray): INDArray = gradient
}
