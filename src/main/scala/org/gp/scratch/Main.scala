package org.gp.scratch

import org.nd4j.linalg.factory.Nd4j

import java.util.Date

//http://yann.lecun.com/exdb/mnist/
//https://github.com/alno/scalann

object Main {
  def main(args: Array[String]): Unit = {
    val x = new InMemoryDataSet("train-images.idx3-ubyte","train-labels.idx1-ubyte")
    val value = x.getEpochIterator(6000)
    while(value.hasNext) {
      value.next()
    }
  }
}
