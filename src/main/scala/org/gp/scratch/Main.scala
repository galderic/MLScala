package org.gp.scratch

import scala.math.tanh

//http://yann.lecun.com/exdb/mnist/
//https://github.com/alno/scalann

object Main {
  def main(args: Array[String]): Unit = {
    val x = new InMemoryDataSet("train-images.idx3-ubyte", "train-labels.idx1-ubyte")

    val dnn = new DNN(new InputLayer(28 * 28, 1000))
    dnn.addFullyConnected(new FullyConnectedLayer(10, f => tanh(f).toFloat))

    val b = x.getEpochIterator(1000).next()
    dnn.fit(b)

//        val value = x.getEpochIterator(6000)
//        var round=0
//        while(value.hasNext) {
//          val b:Batch = value.next()
//          x.saveImage(b,4,s"batch-${round}")
//          round+=1
//        }
  }
}
