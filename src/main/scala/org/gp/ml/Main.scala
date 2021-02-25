package org.gp.ml

import com.typesafe.scalalogging.LazyLogging
import org.gp.ml.dataset.MNISTDataSet
import org.gp.ml.layers.{Activations, FullyConnectedLayer, Softmax}
import org.gp.ml.optimizer.Vanilla
import org.scalameter.measure

object Main extends LazyLogging {
  def main(args: Array[String]): Unit = {

    val trainSet: DataSet = new MNISTDataSet()

    val testSet: DataSet = new MNISTDataSet(true)

    val learningRate = .05d
    val batchSize = 64
    val epochs = 15

    val dnn = new DNN(new SquareLossFunction)
    dnn.addLayer(new FullyConnectedLayer(28 * 28, 150, Vanilla.withLearningRate(learningRate)))
    dnn.addLayer(new Activations.relu)
    dnn.addLayer(new FullyConnectedLayer(150, 10, Vanilla.withLearningRate(learningRate)))
    dnn.addLayer(new Activations.relu)
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
