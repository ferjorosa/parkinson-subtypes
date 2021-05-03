name := "parkinson-subtypes-new"

version := "0.1"

scalaVersion := "2.13.4"

libraryDependencies ++= Seq (
  "colt" % "colt" % "1.2.0",
  "commons-cli" % "commons-cli" % "1.2",
  "org.apache.commons" % "commons-lang3" % "3.6",
  "org.apache.commons" % "commons-math3" % "3.6.1",
  "org.slf4j" % "slf4j-simple" % "1.7.26",
  "com.google.guava" % "guava" % "27.1-jre",
  "com.google.code.gson" % "gson" % "2.8.6"
)