package org.gp.scratch

import org.nd4j.linalg.factory.Nd4j
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should

class SoftmaxTest extends AnyFlatSpec with should.Matchers {

  "Softmax" should "output the expected values in forward pass" in {
    val input = Nd4j.create(Array[Double](1.0, 2.0, 3.0)).reshape(1,3)
    val output = new Softmax().forward(input)
    output.toDoubleVector shouldBe Array[Double](0.09003057317038046,0.24472847105479764,0.6652409557748218)
  }

  "Softmax" should "output the expected values in backward pass" in {
    val input = Nd4j.create(Array[Double](1.0d, 2.0d, 3.0d)).reshape(1,3)
    val sm=new Softmax()
    sm.forward(input)
    val output = sm.backward(input.dup())
    output.toDoubleVector shouldBe Array[Double](-0.14181709360981215d,-0.14077035746963007d,0.2825874510794424d)
  }
}
