package org.gp.scratch

import com.typesafe.scalalogging.LazyLogging
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.ops.transforms.Transforms

object Activations {
  class sigmoid extends Layer with LazyLogging{

    override def forward(inputs: INDArray): INDArray = {
      Transforms.sigmoid(inputs)
    }
    override def backward(gradient: INDArray): INDArray = {
      val result=Transforms.sigmoidDerivative(lastInputs).muli(gradient)
      result
    }
  }

  class relu extends Layer {
    override def forward(inputs: INDArray): INDArray = {
      Transforms.leakyRelu(inputs)
    }
    override def backward(gradient: INDArray): INDArray = {
      val result=Transforms.leakyReluDerivative(lastInputs,0.0d).muli(gradient)
      result
    }
  }
}
