package org.gp.scratch

import com.typesafe.scalalogging.LazyLogging
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

import scala.collection.mutable.ListBuffer

class DNN(val lossFunction: LossFunction) extends LazyLogging {
  def predict(features: INDArray): INDArray = {
    var result = features.div(255).transpose()
    for (layer <- layers) {
      result = layer.forwardPass(result)
    }
    result
  }


  val layers: ListBuffer[Layer] = ListBuffer()

  def addLayer(layer: Layer): Unit = {
    layers.addOne(layer)
  }

  def fit(batch: Batch): Double = {
    var result = batch.features.div(255).transpose()
    for (layer <- layers) {
      result = layer.forwardPass(result)
    }

    val nSamples = result.shape()(0).toInt
    val nOutputs = result.shape()(1).toInt

    val labelVector = Nd4j.zeros(nSamples, nOutputs)
    for (i <- 0 until nSamples) labelVector.putScalar(i, batch.labels.getLong(i), 1)

    val averageLoss = lossFunction.loss(labelVector, result) / nSamples

    var gradient = lossFunction.gradient(labelVector, result)

    for (layer <- layers.reverse) {
      gradient = layer.backwardPass(gradient)
    }

    averageLoss
  }
}
