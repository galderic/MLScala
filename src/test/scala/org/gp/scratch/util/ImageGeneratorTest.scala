package org.gp.scratch.util

import org.gp.ml.dataset.MNISTDataSet
import org.gp.ml.image.ImageGenerator
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should

class ImageGeneratorTest extends AnyFlatSpec with should.Matchers {

  "ImageGenerator" should "generate image from one of the labels" in {
    val x = new MNISTDataSet(true)
    val batch = x.getBatchIterator(100).next()
    ImageGenerator.fromGray(batch, 1, 28, 28, "test")
  }
}
