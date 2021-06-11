package org.gp.ml.layers

import com.typesafe.scalalogging.LazyLogging
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.ops.transforms.Transforms

object Activations {

  class sigmoid(val id: String = "sigmoid") extends Layer with LazyLogging {

    override def forward(inputs: INDArray): INDArray = {
      Transforms.sigmoid(inputs,false)
    }

    override def backward(gradient: INDArray, lastInputs: INDArray): INDArray = {
      Transforms.sigmoidDerivative(lastInputs.muli(gradient), false)
    }
  }

  class leakyRelu(val id: String = "relu") extends Layer {

    override def forward(inputs: INDArray): INDArray = {
      Transforms.leakyRelu(inputs,false)
    }

    override def backward(gradient: INDArray, lastInputs: INDArray): INDArray = {
      // the cutoff is actually the alpha
      Transforms.leakyReluDerivative(lastInputs, 0.01d, false).muli(gradient)
    }
  }

}
