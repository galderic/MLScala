package org.gp.scratch

import org.nd4j.linalg.factory.Nd4j

import java.awt.Color
import java.io.{DataInputStream, File}
import scala.util.Using

class InMemoryDataSet(val samplesFile: String, val labelsFile: String) {

  Using.resources(
    new DataInputStream(getClass.getClassLoader.getResourceAsStream(labelsFile)),
    new DataInputStream(getClass.getClassLoader.getResourceAsStream(samplesFile))
  ) { (labelsStream, samplesStream) =>
    assert(labelsStream.readInt() == 2049, "Wrong MNIST image stream magic")
    assert(samplesStream.readInt() == 2051, "Wrong MNIST image stream magic")

    val (labelsCount, samplesCount) = (labelsStream.readInt(), samplesStream.readInt())
    assert(labelsCount == samplesCount)

    val (width, height) = (samplesStream.readInt(), samplesStream.readInt())
    println(s"width:$width height:$height")
    val bytes = Array.fill[Byte](width * height * samplesCount)(0)

    samplesStream.readFully(bytes)
    val samples = Nd4j.create(bytes.map[Float]{b=>b}, Array(width, height, samplesCount), 'c');

    import java.awt.image.BufferedImage
    val rsm = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY)


    for(x<-1 to height-1) {
      for (y<-1 to width-1) {
        println(samples.getFloat(x,y,1))
        val myWhite = new Color(samples.getFloat(x,y,1), samples.getFloat(x,y,1), samples.getFloat(x,y,1)); // Color white
        rsm.setRGB(x,y,myWhite.getRGB)
      }
    }
    import javax.swing.ImageIcon
    import javax.swing.JPanel

    import javax.imageio.ImageIO
    val outputfile = new File("saved.png")
    ImageIO.write(rsm, "png", outputfile)

    println(samples)
  }
}
