package iet.unipi.bigdatamining.fuzzy.fuzzyset

/**
  * A Fuzzy Set trait to define the common methods
  * that each Fuzzy Set instance should implement.
  */
private[bigdatamining] trait FuzzySet extends Serializable {

  /**
   * The leftmost value of the fuzzy set
   * @return the leftmost value of the fuzzy set
   *          Note that the function returns 0 if xi is Nan
   */
  def left: Double

  /**
   * The rightmost value of the fuzzy set
   * @return the rightmost value of the fuzzy set
   *          Note that the function returns 0 if xi is Nan
   */
  def right: Double
  
  /**
   * The membership degree uA(xi) of the fuzzy set for a given point
   * @param xi the point on which calculate uA 
   * @return the value of membership degree
   *          Note that the function returns 0 if xi is Nan
   */
  def membershipDegree(xi: Double): Double
  
  /**
   * Check if the point belongs to the support of fuzzy set
   * @param xi the point on which check if it belongs to the support of fuzzy set
   * @return true if xi belongs to the support of fuzzy set,
   *          false otherwise or xi is Nan
   */
  def isInSupport(xi: Double): Boolean

  /**
    * The string representing the fuzzy set.
    * For debugging only porpoise.
    */
  def toDebugString: String
  
}