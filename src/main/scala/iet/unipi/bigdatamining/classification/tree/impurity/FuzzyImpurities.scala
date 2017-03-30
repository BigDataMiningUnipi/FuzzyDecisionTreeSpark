package iet.unipi.bigdatamining.classification.tree.impurity

/**
  * Object to create FuzzyImpurity object from an input string.
  * Only accepted name: "fuzzy_entropy"
  */
private[tree] object FuzzyImpurities {

  /**
    * The method creates a FuzzyImpurity object from an input string.
    * Only accepted name: "fuzzy_entropy"
    * Not case-sensitive
    * @param name of the fuzzy entropy
    * @return a FuzzyImpurity instance
    *         FuzzyEntropy in case name equal to fuzzy_entropy (not case-sensitive)
    *         [[java.lang.IllegalArgumentException]] in other all other cases.
    */
  def fromString(name: String): FuzzyImpurity = name.toLowerCase match {
    case "fuzzy_entropy" => FuzzyEntropy
    case _ => throw new IllegalArgumentException(s"Did not recognize Fuzzy Impurity name: $name")
  }
  
}