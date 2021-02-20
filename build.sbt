name := "MLScala"

version := "0.1"

scalaVersion := "2.13.4"

libraryDependencies += "org.nd4j" % "nd4j-cuda-10.1" % "1.0.0-beta7"
libraryDependencies += "org.slf4j" % "slf4j-log4j12" % "1.7.30"

// data fetchers like MnistDataSetIterator
libraryDependencies += "org.deeplearning4j" % "deeplearning4j-core" % "1.0.0-beta7"
libraryDependencies += "com.typesafe.scala-logging" %% "scala-logging" % "3.9.2"
libraryDependencies += "ch.qos.logback" % "logback-classic" % "1.2.3"

libraryDependencies += "org.scalactic" %% "scalactic" % "3.2.2"
libraryDependencies += "org.scalatest" %% "scalatest" % "3.2.2" % "test"
libraryDependencies += "com.storm-enroute" %% "scalameter" % "0.20"