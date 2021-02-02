package org.gp.scratch

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

import scala.collection.mutable.{HashMap, ListBuffer}

class DNN(val inputLayer: InputLayer) {

  val layers: ListBuffer[Layer] = ListBuffer(inputLayer)
  val weights: HashMap[Layer, INDArray] = HashMap.empty

  def addFullyConnected(layer: FullyConnectedLayer): Unit = {
    val w: INDArray = Nd4j.create(layers.last.numOutputs, layer.numUnits)
    layers.addOne(layer)
    weights.addOne(layer, w)
  }

  //https://deeplearning4j.konduit.ai/nd4j/overview
  def fit(batch: Batch): Unit = {
    for (layer <- layers.filter(_.isInstanceOf[HasWeights])) {
      val w:INDArray = weights.get(layer).get
      val x:INDArray = batch.features
      println(s"w shape:${w.shape().mkString(",")} x shape:${x.shape().mkString(",")}")
      val result = x.transpose().mmul(w)
    }
  }
}
