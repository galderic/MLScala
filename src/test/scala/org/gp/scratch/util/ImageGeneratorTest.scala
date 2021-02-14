package org.gp.scratch.util

import org.gp.scratch.InMemoryDataSet
import org.nd4j.linalg.factory.Nd4j
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should

class ImageGeneratorTest extends AnyFlatSpec with should.Matchers {

  "ImageGenerator" should "generate image from one of the labels" in {
    val x = new InMemoryDataSet("train-images.idx3-ubyte", "train-labels.idx1-ubyte")
    val batch = x.getBatchIterator(100).next()
    ImageGenerator.drawBatch(batch,1,28,28,"test")
  }
}
