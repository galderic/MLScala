package org.gp.ml.optimizer

import com.typesafe.scalalogging.LazyLogging
import org.nd4j.linalg.api.ndarray.INDArray

object BasicOptimizer {
  def withLearningRate(learningRate: Double) = new BasicOptimizer(learningRate)
}

protected class BasicOptimizer(val learningRate: Double) extends Optimizer with LazyLogging {

  override def updateWeights(weights: INDArray, gradient: INDArray): Unit = {

    val weightDiff = gradient.mul(learningRate)

    if (!weightDiff.any()) {
      logger.warn("Weights are all zero")
    }

    weights.subi(weightDiff)
  }

  override def name(): String = "Basic Optimizer"
}
