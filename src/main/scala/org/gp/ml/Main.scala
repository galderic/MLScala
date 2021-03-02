package org.gp.ml

import com.typesafe.scalalogging.LazyLogging
import org.gp.ml.dataset.MNISTDataSet
import org.gp.ml.layers.{Activations, FullyConnectedLayer, Softmax}
import org.gp.ml.optimizer.Vanilla
import org.scalameter.measure
import org.tensorflow.framework.Summary
import org.tensorflow.hadoop.util.TFRecordWriter
import org.tensorflow.util.Event

import java.io.{DataOutputStream, FileOutputStream}

object Main extends LazyLogging {


  private def myEvent(name: String): Event = {
    val x = Summary.Value.newBuilder()

    x.setTag(name)
    x.setSimpleValue(0.8f)

    val value = x.build()

    val summary = Summary.newBuilder().addValue(value).build()

    Event.newBuilder()

    Event.newBuilder().setSummary(summary).setStep(0).setWallTime(System.currentTimeMillis()).build()
  }

  def main(args: Array[String]): Unit = {

    val w:TFRecordWriter = new TFRecordWriter(new DataOutputStream(new FileOutputStream("events.out.tfevents.1614294309.v2")))
    w.write(myEvent("epoch_loss").toByteArray)

    val trainSet: DataSet = new MNISTDataSet()
    val testSet: DataSet = new MNISTDataSet(true)

    val learningRate = .07d
    val batchSize = 64
    val epochs = 10

    val dnn = new DNN(new SquareLossFunction)
    dnn.addLayer(new FullyConnectedLayer(28 * 28, 150, Vanilla.withLearningRate(learningRate)))
    dnn.addLayer(new Activations.relu)
    dnn.addLayer(new FullyConnectedLayer(150, 10, Vanilla.withLearningRate(learningRate)))
    dnn.addLayer(new Activations.relu)
    dnn.addLayer(new Softmax)

    val execution_time = measure {

      for (e <- 1 to epochs) {
        val averageLoss = trainSet.getBatchIterator(batchSize).foldLeft(0.0)((_, b) => dnn.fit(b))

        dnn.layers.foreach(_ match {
          case t: Trainable => logger.info(t.summary())
          case _ =>
        }
        )

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
