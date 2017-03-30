package iet.unipi.bigdatamining.fuzzy.fuzzyset

/**
  * The class represents the implementation of a triangular fuzzy set.
  * It implements [[iet.unipi.bigdatamining.fuzzy.fuzzyset.FuzzySet]] trait.
  *
  * The parameters a, b, and c represent the leftmost, the core and the rightmost values of
  * the triangular fuzzy sets.
  *
  * @param a the leftmost value of the triangular fuzzy set
  * @param b the core of the triangular fuzzy set (where membership degree is equal to 1)
  * @param c the rightmost value of the triangular fuzzy set
  */
private[bigdatamining] class TriangularFuzzySet(
    val a: Double,
    val b: Double,
    val c: Double) extends FuzzySet{
  
  private val leftSupportWidth: Double = b - a
  private val rightSupportWidth: Double = c - b
  
  @Override
  def toDebugString: String = {
    s"a=$a, b=$b, c=$c"
  }

  @Override
  def left: Double = a
  
  @Override
  def right: Double = c
  
  @Override
  def isInSupport(xi: Double): Boolean = xi > a && xi < c
  
  @Override
  def membershipDegree(xi: Double): Double = { 
    if (isInSupport(xi)){
      if ((xi <= b && (a == Double.MinValue)) || (xi >= b && (c == Double.MaxValue)))
        1D
      else if (xi <= b)
          (xi - a) / leftSupportWidth
        else
          1D - (xi - b) / rightSupportWidth
    }else
      0D
  }

}

private[bigdatamining] object TriangularFuzzySet extends FuzzySetBuilder {

  @Override
  def createFuzzySet(parameters: Array[Double]): FuzzySet = {
    require(parameters.length == 3, "Triangular Fuzzy Set Builder requires three parameters " +
        s"(left, peak and right), but $parameters.length values have been provided.")
    val sortedParameters = parameters.sorted
    new TriangularFuzzySet(sortedParameters(0), sortedParameters(1), sortedParameters(2)) 
  }
  
  @Override
  def createFuzzySets(points: Array[Double]): Array[FuzzySet] = {
    require(points.length > 1, "Triangular Fuzzy Set Builder requires at least two points, " +
        s"but $points.length points have been provided.")
    createFuzzySetsFromStrongPartition(points.sorted)
  }

  /*
   * Create a strong fuzzy partition starting from an array of points.
   */
  private def createFuzzySetsFromStrongPartition(points: Array[Double]): Array[FuzzySet] = {
    val fuzzySets = new Array[FuzzySet](points.length)
    // Create first fuzzy set
    fuzzySets(0) = new TriangularFuzzySet(Double.MinValue, points(0), points(1))
    // Create middle fuzzy sets
    (1 until points.length-1).foreach { index =>
      fuzzySets(index) = new TriangularFuzzySet(points(index-1), points(index), points(index+1))
    }
    //Create last fuzzy set
    fuzzySets(points.length-1) =
      new TriangularFuzzySet(points(points.length-2), points(points.length-1), Double.MaxValue)

    fuzzySets
  }

}