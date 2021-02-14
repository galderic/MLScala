package org.gp.scratch

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.ops.transforms.Transforms

object Activations {
  class sigmoid extends Layer {
    var lastInputs:INDArray = _

    override def forwardPass(inputs: INDArray): INDArray = {
      this.lastInputs = inputs
      Transforms.sigmoid(inputs)
    }
    override def backwardPass(gradient: INDArray): INDArray = {
      Transforms.sigmoidDerivative(lastInputs).muli(gradient)
    }
  }

  class relu extends Layer {
    var lastInputs:INDArray = _

    override def forwardPass(inputs: INDArray): INDArray = {
      this.lastInputs = inputs
      Transforms.leakyRelu(inputs)
    }
    override def backwardPass(gradient: INDArray): INDArray = {
      val result=Transforms.leakyReluDerivative(lastInputs,0.0d).muli(gradient)
      result
    }
  }
}
