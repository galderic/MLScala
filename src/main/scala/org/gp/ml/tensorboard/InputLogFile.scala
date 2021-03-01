package org.gp.ml.tensorboard

import org.gp.ml.util.LittleEndian
import org.tensorflow.util.Event

import java.io.FileInputStream

class InputLogFile(fileName: String) {

  val fileInputStream = new FileInputStream(fileName)

  def readEvent(): Option[Event] = {
    var length: Long = LittleEndian.littleEndianLong(fileInputStream)

    if (length == 0L)
      return None

    var crc_length = LittleEndian.littleEndianInt(fileInputStream)

    val data = new Array[Byte](length.intValue())

    fileInputStream.read(data)

    val e = Event.parseFrom(data)

    var crc_contents = LittleEndian.littleEndianInt(fileInputStream)

    if (crc_contents != crc32b(data)) {
      throw new IllegalStateException()
    }

    Some(e)
  }

  def crc32b(message: Array[Byte]): Int = {
    org.tensorflow.hadoop.util.Crc32C.maskedCrc32c(message)
  }
}
