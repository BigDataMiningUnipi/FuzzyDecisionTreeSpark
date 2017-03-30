package iet.unipi.bigdatamining.fuzzy.tnorm

/**
  * A TNorm trait to define the common method
  * that each tNorm function  should implement.
  */
private[bigdatamining] trait TNorm extends Serializable {

  /**
   * The result of the T-norm function for the given inputs.
    *
   * @param uAi array of values on which calculate the tNorm
   * @return the value of tNorm, or NaN if at least one value of uAi is NaN
   */
  def calculate(uAi: Double*): Double
  
}