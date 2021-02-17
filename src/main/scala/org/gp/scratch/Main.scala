package org.gp.scratch

import com.typesafe.scalalogging.LazyLogging

object Main extends LazyLogging {
  def main(args: Array[String]): Unit = {
    val trainSet:DataSet = new MNISTDataSet("train-images.idx3-ubyte", "train-labels.idx1-ubyte")

    val testSet:DataSet = new MNISTDataSet("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte")

    val learningRate = .4d
    val batchSize = 128
    val epochs = 10

    val dnn = new DNN(new SquareLossFunction)
    dnn.addLayer(new FullyConnectedLayer(28 * 28, 100, learningRate))
    dnn.addLayer(new Activations.sigmoid)
    dnn.addLayer(new FullyConnectedLayer(100, 10, learningRate))
    dnn.addLayer(new Activations.sigmoid)
    dnn.addLayer(new Softmax)

    for (i <- 0 until epochs) {
      val trainIter = trainSet.getBatchIterator(batchSize)
      while (trainIter.hasNext) {
        val averageLoss = dnn.fit(trainIter.next())
        logger.info(s"Average Loss for batchSize:${batchSize}:${averageLoss}")
      }
    }

    val testIter = testSet.getBatchIterator(testSet.numSamples)

    val testBatch = testIter.next()
    val predictions = dnn.predict(testBatch.features)

    var success = 0.0
    for (i <- 0 until predictions.rows())
      if (predictions.getRow(i).argMax(0).getInt(0) == testBatch.labels.getInt(i))
        success += 1.0

    logger.info(s"Accuracy:${success * 100 / (predictions.rows())}%")
  }
}
