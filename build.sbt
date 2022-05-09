name := "MLScala"

version := "0.1"

scalaVersion := "2.13.8"

libraryDependencies += "org.nd4j" % "nd4j-native" % "1.0.0-M1.1"
libraryDependencies += "org.nd4j" % "nd4j-native" % "1.0.0-M1.1" classifier "linux-x86_64-avx2"

libraryDependencies += "com.typesafe.scala-logging" %% "scala-logging" % "3.9.4"

libraryDependencies += "org.scalactic" %% "scalactic" % "3.2.12"
libraryDependencies += "org.scalatest" %% "scalatest" % "3.2.12" % "test"
libraryDependencies += "com.google.protobuf" % "protobuf-java" % "3.15.3"
libraryDependencies += "org.tensorflow" % "tensorflow-hadoop" % "1.15.0"

Compile / PB.targets := Seq(
  PB.gens.java -> (Compile / sourceManaged).value
)