package org.gp.scratch

import org.nd4j.linalg.factory.Nd4j

import java.util.Date

object Main {
  def main(args: Array[String]): Unit = {
    println("Hello, world!")
    val x = Nd4j.randn(10000,10000)
    val y = Nd4j.randn(10000,10000)

    println(new Date())
    println(x.mmul(y))
    println(new Date())

  }
}
