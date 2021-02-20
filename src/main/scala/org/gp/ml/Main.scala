package org.gp.ml

import com.typesafe.scalalogging.LazyLogging
import org.gp.ml.dataset.MNISTDataSet
import org.scalameter.measure

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

    val execution_time = measure {

      for (e <- 1 to epochs) {
        var averageLoss: Double = 0
        trainSet.getBatchIterator(batchSize).foreach(b => averageLoss = dnn.fit(b))

        dnn.layers.filter(_.isInstanceOf[Trainable]).foreach(trainable => {
          logger.info(trainable.asInstanceOf[Trainable].summary())
        })

        logger.info(s"Average Loss after epoch:$e:$averageLoss")
      }
    }

    val testIter = testSet.getBatchIterator(testSet.numSamples)

    val testBatch = testIter.next()
    val predictions = dnn.predict(testBatch.features)

    val result = ClassifierEval.from(testBatch.labels, predictions)

    logger.info(s"Accuracy after $epochs epochs:$result total time:$execution_time")
  }
}
