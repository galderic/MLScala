name := "MLScala"

version := "0.1"

scalaVersion := "2.13.4"

resolvers +=
  "Sonatype OSS Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots"

libraryDependencies += "org.nd4j" % "nd4j-native" % "1.0.0-beta7"
libraryDependencies += "org.nd4j" % "nd4j-native" % "1.0.0-beta7" classifier "windows-x86_64-avx2"

libraryDependencies += "com.typesafe.scala-logging" %% "scala-logging" % "3.9.2"

libraryDependencies += "org.scalactic" %% "scalactic" % "3.2.2"
libraryDependencies += "org.scalatest" %% "scalatest" % "3.2.2" % "test"
libraryDependencies += "com.storm-enroute" %% "scalameter" % "0.20"
libraryDependencies += "com.google.protobuf" % "protobuf-java" % "3.15.3"
libraryDependencies += "org.tensorflow" % "tensorflow-hadoop" % "1.15.0"

Compile / PB.targets := Seq(
  PB.gens.java -> (Compile / sourceManaged).value
)