package org.gp.ml.dataset

import org.gp.ml
import org.gp.ml.{Batch, DataSet, ImageFeatures}
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex

import java.io.DataInputStream
import scala.util.Random

class MNISTDataSet(val samplesFile: String, val labelsFile: String) extends DataSet with ImageFeatures {

  private val labelsStream = new DataInputStream(getClass.getClassLoader.getResourceAsStream(labelsFile))
  private val samplesStream = new DataInputStream(getClass.getClassLoader.getResourceAsStream(samplesFile))

  assert(labelsStream.readInt() == 2049, "Wrong MNIST image stream magic")
  assert(samplesStream.readInt() == 2051, "Wrong MNIST image stream magic")

  val (numLabels, numSamples) = (labelsStream.readInt(), samplesStream.readInt())
  assert(numLabels == numSamples)

  val (width, height) = (samplesStream.readInt(), samplesStream.readInt())
  private val samplesBytes = new Array[Byte](width * height * numSamples)
  private val labelsBytes = new Array[Byte](numSamples)

  samplesStream.readFully(samplesBytes)
  labelsStream.readFully(labelsBytes)
  private val samples = Nd4j.create(samplesBytes.map[Float] { b => b & 0xff }, Array(width * height, numSamples), 'f');
  private val labels = Nd4j.create(labelsBytes.map[Float] { b => b & 0xff }, Array(numSamples), 'f');

  def getBatchIterator(batchSize: Int): Iterator[Batch] = {
    new Iterator[Batch] {
      var curIndx = 0;
      private val range = 0L until numSamples to List
      val samplesIndx = Random.shuffle(range)

      override def hasNext: Boolean = (curIndx + batchSize) <= numSamples

      override def next(): Batch = {
        val subSamples = samples.get(NDArrayIndex.all(),
          NDArrayIndex.indices(samplesIndx.slice(curIndx, curIndx + batchSize): _*))

        val subLabels = labels.get(NDArrayIndex.indices(samplesIndx.slice(curIndx, curIndx + batchSize): _*))

        curIndx += batchSize
        ml.Batch(subSamples, subLabels)
      }
    }
  }
}