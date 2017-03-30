package iet.unipi.bigdatamining.classification.tree.configuration

/**
  * Object to figure out the Split Strategy enum from an input string.
  * Only accepted name: "binary_split", "multi_split".
  */
object SplitType extends Enumeration {
  type SplitType = Value
  val Binary, Multi = Value

  /**
    * The method initialize a SplitType value from an input string.
    * Only accepted name: "binary_split", "multi_split".
    * Not case-sensitive
    * @param name of the split strategy
    * @return a SplitType value.
    *         Binary in case name equal to binary_split (not case-sensitive)
    *         Multi in case name equal to multi_split (not case-sensitive)
    *         [[java.lang.IllegalArgumentException]] in other all other cases.
    */
  private[tree] def fromString(name: String): SplitType = name.toLowerCase match {
    case "binary_split" => Binary
    case "multi_split" => Multi
    case _ => throw new IllegalArgumentException(s"Did not recognize SplitType name: $name")
  }

}