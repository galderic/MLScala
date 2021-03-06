package org.gp.ml

import com.typesafe.scalalogging.LazyLogging
import org.gp.ml.dataset.{DataSet, MNISTDataSet}
import org.gp.ml.eval.ClassifierEval
import org.gp.ml.layers.{Activations, FullyConnectedLayer, Softmax}
import org.gp.ml.logging.DefaultTracker
import org.gp.ml.loss.SquareLossFunction
import org.gp.ml.optimizer.Vanilla
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.scalameter.measure
import org.tensorflow.framework.Summary
import org.tensorflow.hadoop.util.TFRecordWriter
import org.tensorflow.util.Event

import java.io.{DataOutputStream, FileOutputStream}
import java.nio.file.{Files, Path}
import java.time.Instant
import scala.io.StdIn.readLine

object Main extends LazyLogging {

  def main(args: Array[String]): Unit = {
    logger.info(s"ND4J Data Type Setting: ${Nd4j.dataType()}")

    Files.createDirectories(Path.of("logs"))
    val dos = new DataOutputStream(new FileOutputStream(s"logs/events.out.tfevents.mnist.v2"))
    val w: TFRecordWriter = new TFRecordWriter(dos)

    val trainSet: DataSet = new MNISTDataSet()
    val testSet: DataSet = new MNISTDataSet(true)

    val learningRate = .02d
    val batchSize = (learningRate * 1600).toInt
    logger.info(s"Batch size:$batchSize")
    val epochs = 10

    val dnn = new DNN(new SquareLossFunction, DefaultTracker())
    dnn.addLayer(new FullyConnectedLayer(28 * 28, 100, Vanilla.withLearningRate(learningRate), "fcl_1"))
    dnn.addLayer(new Activations.leakyRelu("activation_1"))
    dnn.addLayer(new FullyConnectedLayer(100, 10, Vanilla.withLearningRate(learningRate), "fcl_2"))
    dnn.addLayer(new Activations.leakyRelu("activation_2"))
    dnn.addLayer(new Softmax)

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
