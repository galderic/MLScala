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
    var result = batch.features.div(255).transpose()
    for (layer <- layers) {
      result = layer.forwardPass(result)
//      println(s"result after:${layer} : $result")
    }

    val nSamples = result.shape()(0).toInt
    val nOutputs = result.shape()(1).toInt

    val labelVector = Nd4j.zeros(nOutputs, nSamples)
    for (i <- 0 until nSamples) labelVector.putScalar(batch.labels.getLong(i), i, 1)

    val averageLoss = lossFunction.cost(result, labelVector) / nSamples

    logger.info(s"Average Loss:${averageLoss}")

    var gradient = lossFunction.derivative(result.transpose(), labelVector)
//    println(s"gradient shape after loss:  ${gradient.shape().mkString(",")}")
//    println(s"gradient after loss:  ${gradient}")

    for (layer <- layers.reverse) {
      gradient = layer.backwardPassWithCheck(gradient)
//      println(s"gradient shape after ${layer}:  ${gradient.shape().mkString(",")}")
//      println(s"gradient after ${layer}:  ${gradient}")
    }

    averageLoss
  }
}
