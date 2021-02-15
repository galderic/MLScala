package org.gp.scratch

import com.typesafe.scalalogging.LazyLogging

object Main extends LazyLogging {
  def main(args: Array[String]): Unit = {
    val trainSet = new InMemoryDataSet("train-images.idx3-ubyte", "train-labels.idx1-ubyte")

    val testSet = new InMemoryDataSet("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte")

    val learningRate = .4d

    val dnn = new DNN(new SquareLossFunction)
    dnn.addLayer(new FullyConnectedLayer(28 * 28, 800, learningRate))
    dnn.addLayer(new Activations.sigmoid)
    dnn.addLayer(new FullyConnectedLayer(800, 10, learningRate))
    dnn.addLayer(new Activations.sigmoid)
    dnn.addLayer(new Softmax)

    val batchSize = 128
    for (i <- 0 until 10) {
      val trainIter = trainSet.getBatchIterator(batchSize)
      while (trainIter.hasNext) {
        val averageLoss = dnn.fit(trainIter.next())
        logger.info(s"Average Loss for batchSize:${batchSize}:${averageLoss}")
      }
    }

    val testIter = testSet.getBatchIterator(testSet.samplesCount)

    val testBatch = testIter.next()
    val predictions = dnn.predict(testBatch.features)

    var success = 0
    for (i <- 0 until predictions.rows())
      if (predictions.getRow(i).argMax(0).getInt(0) == testBatch.labels.getInt(i))
        success += 1.0

    logger.info(s"Accuracy:${success * 100 / (predictions.rows())}%")
  }
}
