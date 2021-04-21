package org.gp.ml

import com.typesafe.scalalogging.{LazyLogging, Logger}
import org.nd4j.linalg.api.ndarray.INDArray

import scala.io.StdIn.readLine

case class DefaultTracker(debug: Boolean = false) extends TrackingCallback with LazyLogging {

  override def afterForward(layer: Layer, inputs: INDArray, outputs: INDArray, batchNum: Int): Unit = {
    if (debug) {
      logger.info(s"Step forward in layer:${layer.id}\n  cachedInputs:${inputs.shape()}$inputs\n  inputGradients:${outputs.shape()}$outputs\n  ${logWeightsIfAvailable(layer)}")
      readLine("Press enter key")
    }
  }

  override def afterBackward(layer: Layer, cachedInputs: INDArray, inputGradient: INDArray, outputGradient: INDArray, batchNum: Int): Unit = {
    if (debug) {
      logger.info(s"Step backward in layer:${layer.id}\n  cachedInputs:${cachedInputs.shape()}$cachedInputs\n  inputGradients:${inputGradient.shape()}$inputGradient\n  \n  ${logWeightsIfAvailable(layer)}")
      readLine("Press enter key")
    }
  }

  override def labelsUsedInCostFunction(labels: INDArray): Unit = {
    if (debug) {
      logger.info(s"Labels${labels.shape()}: $labels)")
      readLine("Press enter key")
    }
  }

  def logWeightsIfAvailable(layer: Layer): String = {
    layer match {
      case t: Trainable =>
        s"weights:{${t.summary()}"
      case _ => ""
    }
  }
}


