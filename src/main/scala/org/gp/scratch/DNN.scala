package org.gp.scratch

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

import scala.collection.mutable.{HashMap, ListBuffer}

class DNN() {

  val layers: ListBuffer[Layer] = ListBuffer()

  def addLayer(layer: Layer): Unit = {
    layers.addOne(layer)
  }

  //https://deeplearning4j.konduit.ai/nd4j/overview
  def fit(batch: Batch): Unit = {
    var result = batch.features
    for (layer <- layers) {
      result = layer.forwardPass(result)
      println(s"Result for layer:$layer is :$result with shape:${result.shape().mkString(",")}")
    }
    println(s"features size:${batch.labels.shape().mkString(",")}")
   val labelVector = Nd4j.zeros(10).putScalar(batch.labels.getLong(0),1)
    println(labelVector)
   println(s"error:${result.distance1(labelVector)}")
  }
}
