package org.gp.scratch

import org.nd4j.linalg.factory.Nd4j

import java.util.Date

//http://yann.lecun.com/exdb/mnist/
//https://github.com/alno/scalann

object Main {
  def main(args: Array[String]): Unit = {
    println("Hello, world!")
//    val x = Nd4j.randn(100,100)
//    val y = Nd4j.randn(100,100)
//
//    val z  = Nd4j.create(1000)
//
//    println(new Date())
//    println(x.mmul(y))
//    println(new Date())

    new InMemoryDataSet("train-images.idx3-ubyte","train-labels.idx1-ubyte")
  }
}
