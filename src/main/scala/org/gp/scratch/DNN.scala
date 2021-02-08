package org.gp.scratch

import com.typesafe.scalalogging.LazyLogging
import org.nd4j.linalg.factory.Nd4j

import scala.collection.mutable.ListBuffer

class DNN(val lossFunction: LossFunction) extends LazyLogging {

  val layers: ListBuffer[Layer] = ListBuffer()

  def addLayer(layer: Layer): Unit = {
    layers.addOne(layer)
  }

  def fit(batch: Batch): Double = {
    var result = batch.features
    for (layer <- layers) result = layer.forwardPass(result)

    val nSamples = result.shape()(0).toInt
    val nOutputs = result.shape()(1).toInt

    val labelVector = Nd4j.zeros(nOutputs, nSamples)
    for (i <- 0 until nSamples) labelVector.putScalar(batch.labels.getLong(i), i, 1)

    val averageLoss = lossFunction.cost(result, labelVector) / nSamples

    logger.info(s"Average Loss:${averageLoss}")

    var gradient = lossFunction.derivative(result.transpose(), labelVector)

    for (layer <- layers) gradient = layer.backwardPass(gradient)

    averageLoss
  }
}
