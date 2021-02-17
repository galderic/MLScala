package org.gp.scratch

trait DataSet {
  def getBatchIterator(batchSize: Int): Iterator[Batch]
  def numSamples:Int
}
