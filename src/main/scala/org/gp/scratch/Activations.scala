package org.gp.scratch

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.ops.transforms.Transforms

object Activations {
  class hardTanh extends Layer {
    override def forwardPass(inputs: INDArray): INDArray = Transforms.hardTanh(inputs)
    override def backwardPass(inputs: INDArray): INDArray =Transforms.hardTanhDerivative(inputs)
  }
}
