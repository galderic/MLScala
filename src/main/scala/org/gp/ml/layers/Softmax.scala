package org.gp.ml.layers

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.ops.impl.transforms.custom.SoftMax
import org.nd4j.linalg.api.ops.impl.transforms.gradient.SoftmaxBp
import org.nd4j.linalg.factory.Nd4j

class Softmax(val id: String = "softmax") extends Layer {

  var cachedResult: Option[INDArray] = None

  override def forward(inputs: INDArray): INDArray = {
    val result = Nd4j.zeros(inputs.rows(), inputs.columns())
    Nd4j.getExecutioner.execAndReturn(new SoftMax(inputs, result, 1))
    result
  }

  override def backward(gradient: INDArray, lastInputs: INDArray): INDArray = {
    if (!cachedResult.isDefined) {
      cachedResult = Some(Nd4j.zeros(lastInputs.rows(), lastInputs.columns()))
    }

    Nd4j.getExecutioner.execAndReturn(new SoftmaxBp(lastInputs, gradient, cachedResult.get, 1))
    cachedResult.get
  }
}
