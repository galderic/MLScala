package org.gp.ml.mnist

import com.typesafe.scalalogging.LazyLogging
import org.gp.ml.DNN
import org.gp.ml.dataset.DataSet
import org.gp.ml.eval.ClassifierEval
import org.gp.ml.event.Utils.myEvent
import org.gp.ml.layers.{Activations, FullyConnectedLayer, Softmax}
import org.gp.ml.logging.DefaultTracker
import org.gp.ml.loss.SquareLossFunction
import org.gp.ml.optimizer.BasicOptimizer
import org.nd4j.linalg.factory.Nd4j
import org.tensorflow.hadoop.util.TFRecordWriter

import java.io.{DataOutputStream, FileOutputStream}
import java.nio.file.{Files, Path}

object FullyConnected extends LazyLogging {

  def main(args: Array[String]): Unit = {
    logger.info(s"ND4J Data Type Setting: ${Nd4j.dataType()}")

    Nd4j.getRandom.setSeed(94834L)

    Files.createDirectories(Path.of("logs"))
    val dos = new DataOutputStream(new FileOutputStream(s"logs/events.out.tfevents.mnist.v2"))
    val w: TFRecordWriter = new TFRecordWriter(dos)

    val trainSet: DataSet = MNISTDataSet.trainDataset()
    val testSet: DataSet = MNISTDataSet.testDataset()

    val learningRate = .02d
    val batchSize = 1
    logger.info(s"Batch size:$batchSize")
    val epochs = 5

    val dnn = DNN.create(SquareLossFunction(), DefaultTracker())
    dnn.addLayer(FullyConnectedLayer(28 * 28, 128, BasicOptimizer.withLearningRate(learningRate), "fcl_1"))
    dnn.addLayer(Activations.leakyRelu("activation_1"))
    dnn.addLayer(FullyConnectedLayer(128, 10, BasicOptimizer.withLearningRate(learningRate), "fcl_2"))
    dnn.addLayer(Activations.leakyRelu("activation_2"))
    dnn.addLayer(Softmax())

    for (e <- 0 until epochs) {
      val averageLoss = trainSet.getBatchIterator(batchSize).foldLeft(0.0)((_, b) => dnn.fit(b))

      logger.info(s"Average Loss after epoch:$e:$averageLoss")
      w.write(myEvent("epoch_loss", averageLoss.floatValue(), e).toByteArray)
      dos.flush()

      val testIter = testSet.getBatchIterator(testSet.numSamples)
      val testBatch = testIter.next()
      val predictions = dnn.predict(testBatch.features)
      val result = ClassifierEval.from(testBatch.labels, predictions)
      logger.info(s"Accuracy after $e epochs:$result")
    }
    dos.close()
  }
}
