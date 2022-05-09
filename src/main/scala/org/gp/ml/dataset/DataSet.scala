package org.gp.ml.dataset

import com.typesafe.scalalogging.LazyLogging

trait DataSet extends LazyLogging {
  def getBatchIterator(batchSize: Int): Iterator[Batch]

  def numSamples: Int

  def width: Int

  def height: Int
}
