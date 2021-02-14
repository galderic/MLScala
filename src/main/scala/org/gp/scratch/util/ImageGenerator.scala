package org.gp.scratch.util

import org.gp.scratch.Batch

import java.awt.Color
import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO

object ImageGenerator {
  def drawBatch(batch: Batch, imageIndex: Int, height: Int, width: Int, prefix: String): Boolean = {
    val rsm = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY)

    for (h <- 0 until height; w <- 0 until width) {
      val g = batch.features.getFloat((h * width + w).toLong, imageIndex) / 255
      val myWhite = new Color(g, g, g);
      rsm.setRGB(w, h, myWhite.getRGB)
    }

    val label = batch.labels.getFloat(imageIndex.toLong)
    ImageIO.write(rsm, "png", new File(s"${prefix}-${label}.png"))
  }
}
