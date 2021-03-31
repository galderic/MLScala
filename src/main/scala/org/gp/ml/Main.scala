package org.gp.ml

import com.typesafe.scalalogging.LazyLogging
import org.gp.ml.dataset.MNISTDataSet
import org.gp.ml.layers.{Activations, FullyConnectedLayer, Softmax}
import org.gp.ml.optimizer.Vanilla
import org.nd4j.linalg.api.ndarray.INDArray
import org.scalameter.measure
import org.tensorflow.framework.Summary
import org.tensorflow.hadoop.util.TFRecordWriter
import org.tensorflow.util.Event

import java.io.{DataOutputStream, FileOutputStream}
import java.nio.file.{Files, Path}
import java.time.Instant
import scala.io.StdIn.readLine

object Main extends LazyLogging {

  private def myEvent(name: String, v: Float, step: Int): Event = {
    val x = Summary.Value.newBuilder()

    x.setTag(name)
    x.setSimpleValue(v)

    val value = x.build()
    val summary = Summary.newBuilder().addValue(value).build()
    Event.newBuilder()

    Event.newBuilder().setSummary(summary).setStep(step).setWallTime(Instant.now.getEpochSecond).build()
  }

  def main(args: Array[String]): Unit = {
    Files.createDirectories(Path.of("logs"))
    val dos = new DataOutputStream(new FileOutputStream(s"logs/events.out.tfevents.mnist.v2"))
    val w: TFRecordWriter = new TFRecordWriter(dos)

    val trainSet: DataSet = new MNISTDataSet()
    val testSet: DataSet = new MNISTDataSet(true)

    val learningRate = .02d
    val batchSize = (learningRate * 1600).toInt //64 // check stdev layer gradients fcl2 64 vs 1024
    val epochs = 20

    def logWeightsIfAvailable(layer: Layer): Unit = {
      layer match {
        case t: Trainable =>
          logger.trace(s"Step forward in Layer:${layer.id} weights:{${t.summary()}")
        case _ =>
      }
    }

    val trackerCallback = new TrackingCallback {
      override def afterForward(layer: Layer, inputs: INDArray, outputs: INDArray, batchNum: Int): Unit = {
        logger.whenTraceEnabled {
          logger.trace(s"Step forward in layer:${layer.id} inputs:${inputs.shape()}$inputs")
          logger.trace(s"Step forward in layer:${layer.id} outputs:${outputs.shape()}$outputs")
          logWeightsIfAvailable(layer)
          readLine("Press enter key")
        }
      }

      override def afterBackward(layer: Layer, cachedInputs: INDArray, inputGradient: INDArray, outputGradient: INDArray, batchNum: Int): Unit = {
        logger.whenTraceEnabled {
          logger.trace(s"Step backward in layer:${layer.id} cachedInputs:${cachedInputs.shape()}$cachedInputs")
          logger.trace(s"Step backward in layer:${layer.id} inputGradients:${inputGradient.shape()}$inputGradient")
          logger.trace(s"Step backward in layer:${layer.id} outputGradients:${outputGradient.shape()}$outputGradient")
          logWeightsIfAvailable(layer)
          readLine("Press enter key")
        }
      }

      override def lossGradient(gradient: INDArray, labels: INDArray): Unit = {
        logger.whenTraceEnabled {
          logger.trace(s"Labels${labels.shape()}: $labels)")
          logger.trace(s"Loss gradient ${gradient.shape()}: $gradient)")
          readLine("Press enter key")
        }
      }

    }

    val dnn = new DNN(new SquareLossFunction, trackerCallback)
    dnn.addLayer(new FullyConnectedLayer(28 * 28, 150, Vanilla.withLearningRate(learningRate), "fcl_1"))
    dnn.addLayer(new Activations.leakyRelu("activation_1"))
    dnn.addLayer(new FullyConnectedLayer(150, 10, Vanilla.withLearningRate(learningRate), "fcl_2"))
    dnn.addLayer(new Activations.leakyRelu("activation_2"))
    dnn.addLayer(new Softmax)

    val execution_time = measure {
      for (e <- 0 until epochs) {
        val averageLoss = trainSet.getBatchIterator(batchSize).foldLeft(0.0)((_, b) => dnn.fit(b))

        logger.info(s"Average Loss after epoch:$e:$averageLoss")
        w.write(myEvent("epoch_loss", averageLoss.floatValue(), e).toByteArray)
        dos.flush()
      }
      dos.close()
    }

    val testIter = testSet.getBatchIterator(testSet.numSamples)

    val testBatch = testIter.next()
    val predictions = dnn.predict(testBatch.features)

    val result = ClassifierEval.from(testBatch.labels, predictions)

    logger.info(s"Accuracy after $epochs epochs:$result total time:$execution_time")
  }
}
