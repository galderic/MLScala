package org.gp.scratch

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

import scala.collection.mutable.LinkedHashMap

class DNN(val numInputs: Int) {

  val weights = LinkedHashMap.empty[Layer, INDArray]

  def addFullyConnected(layer: FullyConnectedLayer): Unit = {
    val w: INDArray = Nd4j.create(weights.last._1.numOutputs, layer.numUnits)
    weights.addOne(layer, w)
  }
}
