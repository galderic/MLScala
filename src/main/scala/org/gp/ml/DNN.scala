package org.gp.ml

import com.typesafe.scalalogging.LazyLogging
import org.gp.ml.dataset.Batch
import org.gp.ml.layers.{CachingLayer, Layer}
import org.gp.ml.logging.TrackingCallback
import org.gp.ml.loss.LossFunction
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

import scala.collection.mutable.ListBuffer

class DNN(val lossFunction: LossFunction, val callback: TrackingCallback) extends LazyLogging {
  val layers: ListBuffer[CachingLayer] = ListBuffer()
  var forward = 0L
  var backward = 0L

  def predict(features: INDArray): INDArray = {
    var result = features.div(255).transpose()
    for (layer <- layers) {
      result = layer.forward(result)
    }
    result
  }

  def addLayer(layer: Layer): Unit = {
    layers.addOne(CachingLayer(layer))
  }

  def fit(batch: Batch): Double = {
    var result = batch.features.div(255).transpose()
    for (layer <- layers) {
      val before = System.currentTimeMillis()
      result = layer.forward(result)
      //if (layer.wrapped.id.equalsIgnoreCase("fcl_1")) {
      forward = forward + System.currentTimeMillis() - before

      //callback.afterForward(layer.wrapped, layer.cachedInputs, result, batch.index)
    }

    val nSamples = result.shape()(0).toInt
    val nOutputs = result.shape()(1).toInt

    val labelVector = Nd4j.zeros(nSamples, nOutputs)
    for (i <- 0 until nSamples) labelVector.putScalar(i, batch.labels.getLong(i), 1)

    val averageLoss = lossFunction.loss(labelVector, result) / nSamples

    var gradient = lossFunction.gradient(labelVector, result)
    //callback.labelsUsedInCostFunction(labelVector)


    for (layer <- layers.reverse) {
      val inputGradients = gradient.dup()
      val before = System.currentTimeMillis()
      gradient = layer.backward(gradient)
      //if (layer.wrapped.id.equalsIgnoreCase("fcl_1"))
      backward += System.currentTimeMillis() - before
      //callback.afterBackward(layer.wrapped, layer.cachedInputs, inputGradients, gradient, batch.index)
    }

    averageLoss
  }
}
