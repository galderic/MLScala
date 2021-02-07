package org.gp.scratch

import org.nd4j.linalg.factory.Nd4j

import scala.collection.mutable.ListBuffer

class DNN() {

  val layers: ListBuffer[Layer] = ListBuffer()

  def addLayer(layer: Layer): Unit = {
    layers.addOne(layer)
  }

  //https://deeplearning4j.konduit.ai/nd4j/overview
  def fit(batch: Batch): Unit = {
    var result = batch.features
    for (layer <- layers) result = layer.forwardPass(result)

    val nSamples = result.shape()(0).toInt
    val nOutputs = result.shape()(1).toInt

    val labelVector = Nd4j.zeros(nOutputs, nSamples)
    for (i <- 0 until nSamples) labelVector.putScalar(batch.labels.getLong(i), i, 1)

    println(s"error:${result.distance1(labelVector)}")
  }
}
