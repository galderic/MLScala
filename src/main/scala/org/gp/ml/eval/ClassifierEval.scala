package org.gp.ml.eval

import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray

object ClassifierEval {
  def from(labels: INDArray, predictions: INDArray): ClassifierEval = {

    val numPredictions = predictions.rows()

    val positives = predictions.argMax(1).castTo(DataType.FLOAT).eq(labels)
      .castTo(DataType.INT16).sum(0).getInt(0).toFloat

    new ClassifierEval(positives * 100 / numPredictions)
  }
}

protected case class ClassifierEval(var accuracy: Float) {
  override def toString: String = {
    s"$accuracy%"
  }
}
