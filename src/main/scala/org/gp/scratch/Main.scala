package org.gp.scratch

import com.typesafe.scalalogging.LazyLogging

object Main extends LazyLogging{
  def main(args: Array[String]): Unit = {
    val trainSet = new InMemoryDataSet("train-images.idx3-ubyte", "train-labels.idx1-ubyte")

    val testSet = new InMemoryDataSet("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte")

    val dnn = new DNN(new SquareLossFunction)
    dnn.addLayer(new FullyConnectedLayer(28 * 28, 800))
    dnn.addLayer(new Activations.sigmoid)
    dnn.addLayer(new FullyConnectedLayer(800, 10))
    dnn.addLayer(new Activations.sigmoid)

    dnn.addLayer(new Softmax)

    for (i <- 0 until 10) {
      val trainIter = trainSet.getBatchIterator(128)
      while (trainIter.hasNext) dnn.fit(trainIter.next())
    }

    val testIter = testSet.getBatchIterator(testSet.samplesCount)

    val testBatch = testIter.next()
    val predictions = dnn.predict(testBatch.features)
    var success,fail=0.0
    for (i<-0 until predictions.rows()) {
//      println(predictions.getRow(i))
//      println(testBatch.labels.getInt(i))
//      println(predictions.getRow(i).argMax(0))
      if (predictions.getRow(i).argMax(0).getInt(0)==testBatch.labels.getInt(i)) {
        success+=1.0
      } else {
        fail+=1.0
      }
    }
    logger.info(s"Accuracy:${success*100/(success+fail)}%")
  }
}
