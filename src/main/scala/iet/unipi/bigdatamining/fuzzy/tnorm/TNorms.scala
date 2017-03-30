package iet.unipi.bigdatamining.fuzzy.tnorm

/**
  * Object to create TNorm object from an input string.
  * Only accepted name: "min", "product".
  */
private[bigdatamining] object TNorms {

  /**
    * The method creates a tNorm from an input string.
    * Only accepted name: "min", "product".
    * Not case-sensitive
    * @param name of the tNorm
    * @return a tNorm instance.
    *         MinTNorm in case name equal to min (not case-sensitive)
    *         ProductTNorm in case name equal to product (not case-sensitive)
    *         [[java.lang.IllegalArgumentException]] in other all other cases.
    */
  def fromString(name: String): TNorm = name.toLowerCase match {
    case "min" => MinTNorm
    case "product" => ProductTNorm
    case _ => throw new IllegalArgumentException(s"Did not recognize t-norm name: $name")
  }
  
}