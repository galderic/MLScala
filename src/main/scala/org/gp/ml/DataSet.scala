package org.gp.ml

trait DataSet {
  def getBatchIterator(batchSize: Int): Iterator[Batch]

  def numSamples: Int
}
