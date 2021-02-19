package org.gp.scratch

import com.typesafe.scalalogging.LazyLogging
import org.nd4j.linalg.api.buffer.DataType

import java.time.Duration

object Main extends LazyLogging {
  def main(args: Array[String]): Unit = {

    val trainSet: DataSet = new MNISTDataSet("train-images.idx3-ubyte", "train-labels.idx1-ubyte")

    val testSet: DataSet = new MNISTDataSet("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte")

    val learningRate = .4d
    val batchSize = 128
    val epochs = 8


    val dnn = new DNN(new SquareLossFunction)
    dnn.addLayer(new FullyConnectedLayer(28 * 28, 100, learningRate))
    dnn.addLayer(new Activations.sigmoid)
    dnn.addLayer(new FullyConnectedLayer(100, 10, learningRate))
    dnn.addLayer(new Activations.sigmoid)
    dnn.addLayer(new Softmax)

    val start = System.currentTimeMillis()
    for (e <- 1 to epochs) {
      var averageLoss: Double = 0
      trainSet.getBatchIterator(batchSize).foreach(b => averageLoss = dnn.fit(b))
      logger.info(s"Average Loss after epoch:$e:$averageLoss")
      dnn.layers.filter(_.isInstanceOf[Trainable]).foreach(trainable => {
        logger.info(trainable.asInstanceOf[Trainable].summary())
      })
    }

    val testIter = testSet.getBatchIterator(testSet.numSamples)

    val testBatch = testIter.next()
    val predictions = dnn.predict(testBatch.features)

    var success = 0.0
    for (i <- 0 until predictions.rows())
      if (predictions.getRow(i).argMax(0).getInt(0) == testBatch.labels.getInt(i))
        success += 1.0

    val end=System.currentTimeMillis()
    logger.info(s"Accuracy after $epochs epochs:${success * 100 / predictions.rows()}% total time:${Duration.ofMillis(end-start).toSeconds} seconds")
  }
}
