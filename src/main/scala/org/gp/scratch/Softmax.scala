package org.gp.scratch

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.ops.transforms.Transforms

class Softmax() extends Layer {
  override def forwardPass(inputs: INDArray): INDArray = {
//    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
//    return e_x / np.sum(e_x, axis=-1, keepdims=True)
    Transforms.softmax(inputs)
  }
}
