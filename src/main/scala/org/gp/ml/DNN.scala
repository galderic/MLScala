package org.gp.ml

import com.typesafe.scalalogging.LazyLogging
import org.gp.ml.dataset.Batch
import org.gp.ml.layers.{CachingLayer, Layer}
import org.gp.ml.logging.{DefaultTracker, NoOpTracker, Tracker}
import org.gp.ml.loss.LossFunction
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

import scala.collection.mutable.ListBuffer

object DNN {
  def create(lossFunction: LossFunction): DNN = {
    new DNN(lossFunction, NoOpTracker())
  }

  def create(lossFunction: LossFunction, tracker: Tracker): DNN = {
    new DNN(lossFunction, tracker)
  }
}

class DNN private(lossFunction: LossFunction, callback: Tracker) extends LazyLogging {
  val layers: ListBuffer[CachingLayer] = ListBuffer()

  def predict(features: INDArray): INDArray = {
    var result = features.transpose()
    for (layer <- layers) {
      result = layer.forward(result)
    }
    result
  }

  def addLayer(layer: Layer): Unit = {
    layers.addOne(CachingLayer(layer))
  }

  def fit(batch: Batch): Double = {
    var result = batch.features.transpose()
    for (layer <- layers) {
      result = layer.forward(result)
      callback.afterForward(layer.wrapped, layer.cachedInputs, result, batch.index)
    }

    val nSamples = result.shape()(0).toInt
    val nOutputs = result.shape()(1).toInt

    val labelVector = Nd4j.zeros(nSamples, nOutputs)
    for (i <- 0 until nSamples) labelVector.putScalar(i, batch.labels.getLong(i), 1)

    val averageLoss = lossFunction.loss(labelVector, result) / nSamples

    var gradient = lossFunction.gradient(labelVector, result)
    callback.labelsUsedInCostFunction(labelVector)
    callback.lossFunctionGradient(gradient)

    for (layer <- layers.reverse) {
      val inputGradients = gradient.dup()
      gradient = layer.backward(gradient)
      callback.afterBackward(layer.wrapped, layer.cachedInputs, inputGradients, gradient, batch.index)
    }

    averageLoss
  }
}
