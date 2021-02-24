package org.gp.ml.optimizer

import com.typesafe.scalalogging.LazyLogging
import org.gp.ml.Optimizer
import org.nd4j.linalg.api.ndarray.INDArray

object Vanilla {
  def withLearningRate(learningRate: Double) = new Vanilla(learningRate)
}

protected class Vanilla(val learningRate: Double) extends Optimizer with LazyLogging {

  override def updateWeights(weights: INDArray, gradient: INDArray): Unit = {

    val weightDiff = gradient.mul(learningRate)

    if (!weightDiff.any()) {
      logger.warn("Weights are all zero")
    }

    weights.subi(weightDiff)
  }
}
