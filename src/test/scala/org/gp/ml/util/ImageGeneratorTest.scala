package org.gp.ml.util

import org.gp.ml.dataset.MNISTDataSet
import org.gp.ml.image.ImageGenerator
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should

class ImageGeneratorTest extends AnyFlatSpec with should.Matchers {

  "ImageGenerator" should "generate image from one of the labels" in {
    val mnist = new MNISTDataSet(true)
    val batch = mnist.getBatchIterator(1).next()
    ImageGenerator.fromGray(batch, 0, mnist.height, mnist.width, "test")
  }
}
