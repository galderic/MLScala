package org.gp.ml.mnist

import com.typesafe.scalalogging.LazyLogging
import org.gp.ml.dataset.{Batch, DataSet}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex

import java.io.{DataInputStream, File, FileInputStream}
import java.net.URL
import java.nio.file.{Files, Paths}
import java.util.zip.GZIPInputStream
import scala.collection.mutable
import scala.language.postfixOps
import scala.sys.process._
import scala.util.Random
import scala.util.Using

object MNISTDataSet extends LazyLogging {

  val mnistTempFolder = s"${System.getProperty("""java.io.tmpdir""")}${File.separator}mnist"
  Files.createDirectories(Paths.get(mnistTempFolder))
  private val downloadUrl = "https://github.com/fgnt/mnist/raw/master"

  def trainDataset(): DataSet = {
    ds("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz")
  }

  def testDataset(): DataSet = {
    ds("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz")
  }


  private def downloadIfNecessary(filename: String) = {
    val fullPathName = s"$mnistTempFolder/$filename"
    if (!Files.exists(Paths.get(fullPathName))) {
      new URL(s"$downloadUrl/$filename") #> new File(fullPathName) !!
    }
  }

  private def ds(imagesFilename: String, labelsFilename: String): MNISTDataSet = {

    downloadIfNecessary(imagesFilename)
    downloadIfNecessary(labelsFilename)

    Using.resources(
      new DataInputStream(fromCompressedStream(imagesFilename)),
      new DataInputStream(fromCompressedStream(labelsFilename))) {
      (samplesStream, labelsStream) =>
        assert(labelsStream.readInt() == 2049, "Wrong MNIST image stream magic")
        assert(samplesStream.readInt() == 2051, "Wrong MNIST image stream magic")

        val numLabels = labelsStream.readInt()
        val numSamples = samplesStream.readInt()
        assert(numLabels == numSamples)

        val width = samplesStream.readInt()
        val height = samplesStream.readInt()
        val samplesBytes = new Array[Byte](width * height * numSamples)
        val labelsBytes = new Array[Byte](numSamples)

        samplesStream.readFully(samplesBytes)
        labelsStream.readFully(labelsBytes)
        val samples = Nd4j.create(samplesBytes.map[Float] { b => (b & 0xff) / 255.0f }, Array(width * height, numSamples), 'f')
        val labels = Nd4j.create(labelsBytes.map[Float] { b => b & 0xff }, Array(numSamples), 'f')

        MNISTDataSet(samples, labels, numSamples, width, height)
    }
  }

  private def fromCompressedStream(fileName: String): DataInputStream = {
    val absoluteFileName = s"$mnistTempFolder${File.separator}$fileName"
    logger.info(s"Reading from $absoluteFileName")
    new DataInputStream(new GZIPInputStream(new FileInputStream(absoluteFileName)))
  }
}

case class MNISTDataSet(samples: INDArray, labels: INDArray, numSamples: Int, width: Int, height: Int) extends DataSet {

  override def getBatchIterator(batchSize: Int): Iterator[Batch] = {
    new Iterator[Batch] {
      var curIndx = 0
      private val range = 0L until numSamples to List
      val samplesIndx: mutable.Buffer[Long] = Random.shuffle(range).toBuffer

      override def hasNext: Boolean = (curIndx + batchSize) <= numSamples

      override def next(): Batch = {

        val elements = scala.collection.mutable.ListBuffer.empty[Long]

        for (i <- curIndx until curIndx + batchSize) {
          elements.addOne(samplesIndx(i))
        }

        val ndIndices = NDArrayIndex.indices(elements.toArray: _*)

        val subSamples = samples.get(NDArrayIndex.all(), ndIndices)

        val subLabels = labels.get(ndIndices)

        val result = Batch(subSamples, subLabels, curIndx / batchSize)
        curIndx += batchSize
        result
      }
    }
  }
}