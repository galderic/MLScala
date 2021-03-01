package org.gp.ml.tensorboard

import org.gp.ml.util.LittleEndian
import org.tensorflow.util.Event

import java.io.FileOutputStream

class OutputLogFile(fileName: String) {

  val fileOutputStream = new FileOutputStream(fileName)

  def writeEvent(event: Event): Unit = {
    val serialized = event.toByteArray
    fileOutputStream.write(LittleEndian.toLittleEndian(serialized.length.longValue()))

    fileOutputStream.write(LittleEndian.toLittleEndian(crc32b(LittleEndian.toLittleEndian(serialized.length.longValue()))))
    fileOutputStream.write(serialized)

    fileOutputStream.write(LittleEndian.toLittleEndian(crc32b(serialized)))
    fileOutputStream.flush()
  }

  def crc32b(message: Array[Byte]): Int = {
    org.tensorflow.hadoop.util.Crc32C.maskedCrc32c(message)
  }
}
