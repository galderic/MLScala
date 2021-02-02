package org.gp.scratch

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

import scala.collection.mutable.{HashMap, ListBuffer}

class DNN() {

  val layers: ListBuffer[Layer] = ListBuffer()
//  val weights: HashMap[Layer, INDArray] = HashMap.empty

  def addFullyConnected(layer: FullyConnectedLayer): Unit = {
    layers.addOne(layer)
  }

  //https://deeplearning4j.konduit.ai/nd4j/overview
  def fit(batch: Batch): Unit = {
    for (layer <- layers.filter(_.isInstanceOf[TrainableLayer])) {
      println(layer.forwardPass(batch.features))
    }
  }
}
