package org.gp.scratch

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.ops.impl.shape.OneHot
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex

import java.io.DataInputStream
import scala.util.Random

class InMemoryDataSet(val samplesFile: String, val labelsFile: String) {

  private val labelsStream = new DataInputStream(getClass.getClassLoader.getResourceAsStream(labelsFile))
  private val samplesStream = new DataInputStream(getClass.getClassLoader.getResourceAsStream(samplesFile))

  assert(labelsStream.readInt() == 2049, "Wrong MNIST image stream magic")
  assert(samplesStream.readInt() == 2051, "Wrong MNIST image stream magic")

  val (labelsCount, samplesCount) = (labelsStream.readInt(), samplesStream.readInt())
  assert(labelsCount == samplesCount)

  private val (width, height) = (samplesStream.readInt(), samplesStream.readInt())
  private val samplesBytes = new Array[Byte](width * height * samplesCount)
  private val labelsBytes = new Array[Byte](samplesCount)

  samplesStream.readFully(samplesBytes)
  labelsStream.readFully(labelsBytes)
  private val samples = Nd4j.create(samplesBytes.map[Float] { b => b & 0xff }, Array(width * height, samplesCount), 'f');
  private val labels = Nd4j.create(labelsBytes.map[Float] { b => b & 0xff }, Array(samplesCount), 'f');

  val labelVector: INDArray = Nd4j.create(labelsCount, 10)
  Nd4j.getExecutioner.execAndReturn(new OneHot(labels, labelVector, 10, 1, 1.0d, 0.0d))


  def getBatchIterator(batchSize: Int): Iterator[Batch] = {
    new Iterator[Batch] {
      var curIndx = 0;
      private val range = 0L until samplesCount to List
      val samplesIndx = Random.shuffle(range)

      override def hasNext: Boolean = (curIndx + batchSize) <= samplesCount

      override def next(): Batch = {
        val subSamples = samples.get(NDArrayIndex.all(),
          NDArrayIndex.indices(samplesIndx.slice(curIndx, curIndx + batchSize): _*))

        val subLabels = labels.get(NDArrayIndex.indices(samplesIndx.slice(curIndx, curIndx + batchSize): _*))

        curIndx += batchSize
        Batch(subSamples, subLabels)
      }
    }
  }
}