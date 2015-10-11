version := "1.0.0"

name := "iris"

resolvers += "Sonatype Public" at "https://oss.sonatype.org/content/groups/public/"

scalaVersion := "2.11.7"

libraryDependencies ++= cats ++ dl4j

lazy val cats = Seq(
  "org.spire-math" %% "cats" % "0.2.0"
)

lazy val dl4j = Seq(
  "org.deeplearning4j" % "deeplearning4j-core" % "0.4-rc3.4",
  "org.nd4j"           % "nd4j-x86"               % "0.4-rc3.5"
)