package org.gp.scratch

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

class FullyConnectedLayer(val numUnits: Int, val activationFunction: Float => Float) extends Layer with HasWeights with HasActivation {

  val numOutputs = numUnits

  override def createWeights(inputs: Int): INDArray = {
    Nd4j.create(inputs, numUnits)
  }
}
