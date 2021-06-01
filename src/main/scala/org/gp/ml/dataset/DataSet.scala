package org.gp.ml.dataset

import com.typesafe.scalalogging.LazyLogging

import java.io.File
import java.net.URL
import scala.language.postfixOps
import scala.sys.process._

trait DataSet extends LazyLogging {
  def getBatchIterator(batchSize: Int): Iterator[Batch]

  def numSamples: Int

  def fileDownloader(url: String, filename: String): Unit = {
    logger.info(s"Downloading $url")
    new URL(url) #> new File(filename) !!
  }
}
