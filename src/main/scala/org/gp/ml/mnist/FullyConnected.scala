package org.gp.ml.mnist

import com.typesafe.scalalogging.LazyLogging
import org.gp.ml.DNN
import org.gp.ml.dataset.DataSet
import org.gp.ml.eval.ClassifierEval
import org.gp.ml.layers.{Activations, FullyConnectedLayer, Softmax}
import org.gp.ml.loss.SquareLossFunction
import org.gp.ml.optimizer.BasicOptimizer
import org.nd4j.linalg.factory.Nd4j
import org.tensorflow.framework.Summary
import org.tensorflow.hadoop.util.TFRecordWriter
import org.tensorflow.util.Event

import java.io.{DataOutputStream, FileOutputStream}
import java.nio.file.{Files, Path}
import java.time.Instant

object FullyConnected extends LazyLogging {

  def main(args: Array[String]): Unit = {
    logger.info(s"ND4J Data Type Setting: ${Nd4j.dataType()}")

    Files.createDirectories(Path.of("logs"))
    val dos = new DataOutputStream(new FileOutputStream(s"logs/events.out.tfevents.mnist.v2"))
    val w: TFRecordWriter = new TFRecordWriter(dos)

    val trainSet: DataSet = MNISTDataSet.trainDataset()
    val testSet: DataSet = MNISTDataSet.testDataset()

    val learningRate = .02d
    val batchSize = (learningRate * 1600).toInt
    logger.info(s"Batch size:$batchSize")
    val epochs = 10

    val dnn = DNN.create(SquareLossFunction())
    dnn.addLayer(FullyConnectedLayer(28 * 28, 50, BasicOptimizer.withLearningRate(learningRate), "fcl_1"))
    dnn.addLayer(Activations.leakyRelu("activation_1"))
    dnn.addLayer(FullyConnectedLayer(50, 10, BasicOptimizer.withLearningRate(learningRate), "fcl_2"))
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

  private def myEvent(name: String, v: Float, step: Int): Event = {
    val x = Summary.Value.newBuilder()

    x.setTag(name)
    x.setSimpleValue(v)

    val value = x.build()
    val summary = Summary.newBuilder().addValue(value).build()
    Event.newBuilder()

    Event.newBuilder().setSummary(summary).setStep(step).setWallTime(Instant.now.getEpochSecond).build()
  }
}
