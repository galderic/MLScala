package org.gp.ml.logging

import com.typesafe.scalalogging.LazyLogging
import org.gp.ml.layers.{Layer, Trainable}
import org.nd4j.linalg.api.ndarray.INDArray

import scala.io.StdIn.readLine

case class DefaultTracker() extends Tracker with LazyLogging {

  override def afterForward(layer: Layer, inputs: INDArray, outputs: INDArray, batchNum: Int): Unit = {
    logger.info(s"After forward in layer:${layer.id}\n  inputs:${summary(inputs)}\n  outputs:${summary(outputs)}\n  ${logWeightsIfAvailable(layer)}")
    readLine("Press enter key")
  }

  override def afterBackward(layer: Layer, cachedInputs: INDArray, inputGradient: INDArray, outputGradient: INDArray, batchNum: Int): Unit = {
    logger.info(s"After backward in layer:${layer.id}\n  cachedInputs:${summary(cachedInputs)}\n  inputGradients:${summary(inputGradient)}\n  \n  ${logWeightsIfAvailable(layer)}")
    readLine("Press enter key")
  }

  private def logWeightsIfAvailable(layer: Layer): String = {
    layer match {
      case t: Trainable =>
        s"weights:{${t.summary()}"
      case _ => ""
    }
  }

  private def summary(array: INDArray): String = {
    s"[${array.shape().mkString(",")}] mean:${array.mean()} std:${array.std()} contents:$array"
  }

  override def labelsUsedInCostFunction(labels: INDArray): Unit = {
    logger.info(s"Labels${labels.shape()}: $labels)")
    readLine("Press enter key")
  }

  override def lossFunctionGradient(gradients: INDArray): Unit = {
    logger.info(s"Loss function gradient${gradients.shape()}: $gradients")
    readLine("Press enter key")
  }
}


