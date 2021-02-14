package org.gp.scratch

import com.typesafe.scalalogging.LazyLogging
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.ops.transforms.Transforms

object Activations {
  class sigmoid extends Layer with LazyLogging{
    var lastInputs:INDArray = _

    override def forwardPass(inputs: INDArray): INDArray = {
      this.lastInputs = inputs
      Transforms.sigmoid(inputs)
    }
    override def backwardPass(gradient: INDArray): INDArray = {
      val result=Transforms.sigmoidDerivative(lastInputs).muli(gradient)
//      logger.info(s"sigmoid backward pass average:${result.amean(0)}")
      result
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
