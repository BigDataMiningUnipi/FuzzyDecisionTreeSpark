package iet.unipi.bigdatamining.fuzzy.fuzzyset

/**
  * Object to create a Fuzzy Set object from an input string.
  * Only accepted name: "triangular", "singleton".
  */
private[bigdatamining] object FuzzySets {

  /**
    * The method creates a Fuzzy Set from an input string.
    * Only accepted name: "triangular", "singleton".
    * Not case-sensitive
    * @param name of the Fuzzy Set
    * @return a Fuzzy Set instance.
    *         TriangularFuzzySet in case name equal to triangular (not case-sensitive)
    *         SingletonFuzzySet in case name equal to singleton (not case-sensitive)
    *         [[java.lang.IllegalArgumentException]] in other all other cases.
    */
  def fromString(name: String): FuzzySetBuilder = name.toLowerCase match {
    case "triangular" => TriangularFuzzySet
    case "singleton" => SingletonFuzzySet
    case _ => throw new IllegalArgumentException(s"Did not recognize fuzzy set name: $name")
  }
}