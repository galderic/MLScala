package org.gp.ml.layers

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.ops.impl.transforms.custom.SoftMax
import org.nd4j.linalg.api.ops.impl.transforms.gradient.SoftmaxBp
import org.nd4j.linalg.factory.Nd4j

case class Softmax(id: String = "softmax") extends Layer {
  override def forward(inputs: INDArray): INDArray = {
    val result = Nd4j.zeros(inputs.rows(), inputs.columns())
    Nd4j.getExecutioner.execAndReturn(new SoftMax(inputs, result, 1))
    result
  }

  override def backward(gradient: INDArray, lastInputs: INDArray): INDArray = {
    val result = Nd4j.zeros(lastInputs.rows(), lastInputs.columns()).detach()
    Nd4j.getExecutioner.execAndReturn(new SoftmaxBp(lastInputs, gradient, result, 1))
    result
  }
}
