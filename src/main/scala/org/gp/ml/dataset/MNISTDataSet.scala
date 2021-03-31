package org.gp.ml.dataset

import org.gp.ml
import org.gp.ml.{Batch, DataSet, ImageFeatures}
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex

import java.io.{DataInputStream, File, FileInputStream}
import java.nio.file.{Files, Paths}
import java.util.zip.GZIPInputStream
import scala.util.Random

class MNISTDataSet(val test: Boolean = false) extends DataSet with ImageFeatures {

  private val downloadUrl = "https://github.com/fgnt/mnist/raw/master"

  val mnistTempFolder = s"${System.getProperty("""java.io.tmpdir""")}${File.separator}mnist"
  Files.createDirectories(Paths.get(mnistTempFolder))

  private val files = Map(
    "trainImages" -> "train-images-idx3-ubyte.gz",
    "trainLabels" -> "train-labels-idx1-ubyte.gz",
    "testImages" -> "t10k-images-idx3-ubyte.gz",
    "testLabels" -> "t10k-labels-idx1-ubyte.gz"
  )

  files.values.foreach(f => {
    val target = s"$mnistTempFolder/$f"
    if (!Files.exists(Paths.get(target))) {
      fileDownloader(s"$downloadUrl/$f", target)
    }
  })


  def fromCompressedStream(fileName: String): DataInputStream = {
    val absoluteFileName = s"$mnistTempFolder${File.separator}$fileName"
    logger.info(s"Reading from $absoluteFileName")
    new DataInputStream(new GZIPInputStream(new FileInputStream(absoluteFileName)))
  }

  private val (samplesStream, labelsStream) = {
    if (test) {
      (new DataInputStream(fromCompressedStream(files.get("testImages").get)),
        new DataInputStream(fromCompressedStream(files.get("testLabels").get)))
    } else {
      (new DataInputStream(fromCompressedStream(files.get("trainImages").get)),
        new DataInputStream(fromCompressedStream(files.get("trainLabels").get)))
    }
  }

  assert(labelsStream.readInt() == 2049, "Wrong MNIST image stream magic")
  assert(samplesStream.readInt() == 2051, "Wrong MNIST image stream magic")

  val (numLabels, numSamples) = (labelsStream.readInt(), samplesStream.readInt())
  assert(numLabels == numSamples)

  val (width, height) = (samplesStream.readInt(), samplesStream.readInt())
  private val samplesBytes = new Array[Byte](width * height * numSamples)
  private val labelsBytes = new Array[Byte](numSamples)

  samplesStream.readFully(samplesBytes)
  labelsStream.readFully(labelsBytes)
  private val samples = Nd4j.create(samplesBytes.map[Float] { b => b & 0xff }, Array(width * height, numSamples), 'f')
  private val labels = Nd4j.create(labelsBytes.map[Float] { b => b & 0xff }, Array(numSamples), 'f')

  samplesStream.close()
  labelsStream.close()

  def getBatchIterator(batchSize: Int): Iterator[Batch] = {
    new Iterator[Batch] {
      var curIndx = 0
      private val range = 0L until numSamples to List
      val samplesIndx: List[Long] = Random.shuffle(range)

      override def hasNext: Boolean = (curIndx + batchSize) <= numSamples

      override def next(): Batch = {
        val subSamples = samples.get(NDArrayIndex.all(),
          NDArrayIndex.indices(samplesIndx.slice(curIndx, curIndx + batchSize): _*))

        val subLabels = labels.get(NDArrayIndex.indices(samplesIndx.slice(curIndx, curIndx + batchSize): _*))

        val result = ml.Batch(subSamples, subLabels, curIndx / batchSize)
        curIndx += batchSize
        result
      }
    }
  }
}