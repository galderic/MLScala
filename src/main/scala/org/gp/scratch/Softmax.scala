package org.gp.scratch

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.ops.transforms.Transforms
import org.nd4j.linalg.api.ops.impl
import org.nd4j.linalg.api.ops.impl.transforms.custom.SoftMax
import org.nd4j.linalg.api.ops.impl.transforms.gradient.SoftmaxBp
import org.nd4j.linalg.factory.Nd4j

class Softmax() extends Layer {
  var lastInputs:INDArray = _
  override def forwardPass(inputs: INDArray): INDArray = {
    lastInputs = inputs
    val result = inputs.dup()
    Nd4j.getExecutioner.execAndReturn(new SoftMax(inputs, result,1))
    result
  }

  override def backwardPass(gradient: INDArray): INDArray = {
    val result=lastInputs.dup()
    Nd4j.getExecutioner.execAndReturn(new SoftmaxBp(lastInputs,gradient,result,1))
    result
  }
}
