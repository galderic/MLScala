package org.gp.scratch

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

trait TrainableLayer extends Layer {
  val weights: INDArray = Nd4j.create(numInputs, numOutputs)
}
