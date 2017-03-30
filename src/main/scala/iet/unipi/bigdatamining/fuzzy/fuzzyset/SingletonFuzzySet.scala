package iet.unipi.bigdatamining.fuzzy.fuzzyset

/**
  * The class represents the implementation of a singleton fuzzy set.
  * It implements [[iet.unipi.bigdatamining.fuzzy.fuzzyset.FuzzySet]] trait.
  *
  * The parameters value represents the unique values of
  * the singleton fuzzy sets. It defines also the both the core and support
  *
  * @param value the value of the singleton fuzzy set
  */
private[bigdatamining] class SingletonFuzzySet(
    val value : Double) extends FuzzySet{

  private implicit def bool2int(b:Boolean): Double = if (b) 1D else 0D

  @Override
  def toDebugString: String = {
    s"value=$value"
  }
    
  @Override
  def left: Double = value
  
  @Override
  def right: Double = value
  
  @Override
  def isInSupport(xi: Double): Boolean = xi == value

  @Override
  def membershipDegree(xi: Double): Double = isInSupport(xi).toDouble
  
}

private[bigdatamining] object SingletonFuzzySet extends FuzzySetBuilder {
    
  @Override
  def createFuzzySet(parameters: Array[Double]): FuzzySet = {
    require(parameters.length == 1, "Invalid parameter for Singleton Fuzzy Set. " +
      s"At least one parameter is needed")
    new SingletonFuzzySet(parameters(0)) 
  }
  
  @Override
  def createFuzzySets(points: Array[Double]): Array[FuzzySet] = {
    points.sorted.map(point => createFuzzySet(Array(point)))
  }
}