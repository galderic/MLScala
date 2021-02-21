package org.gp.ml

import com.typesafe.scalalogging.LazyLogging

import sys.process._
import java.net.URL
import java.io.File

trait DataSet extends LazyLogging{
  def getBatchIterator(batchSize: Int): Iterator[Batch]

  def numSamples: Int

  def fileDownloader(url: String, filename: String):Unit = {
    logger.info(s"Downloading $url")
    new URL(url) #> new File(filename) !!
  }
}
