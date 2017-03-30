package iet.unipi.bigdatamining.fuzzy.fuzzyset

/**
  * Fuzzy Set builder trait for creating fuzzy partitions.
  * Each Fuzzy Set object should implement such a trait
  */
private[bigdatamining] trait FuzzySetBuilder extends Serializable{

  /**
   * Create a Fuzzy Set from a set of parameters
   * @param parameters array of parameters
   * @return a Fuzzy Set instance
   */
  def createFuzzySet(parameters: Array[Double]): FuzzySet

  /**
    * Create a bunch of Fuzzy Sets from a set of points
    * @param points array of points (should be within the domain of the feature)
    * @return an array of Fuzzy Set instances
    */
  def createFuzzySets(points: Array[Double]): Array[FuzzySet]
  
}