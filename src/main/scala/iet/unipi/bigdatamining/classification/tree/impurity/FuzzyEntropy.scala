package iet.unipi.bigdatamining.classification.tree.impurity

/**
  * The class implements the fuzzy entropy logic.
  * It implement the [[iet.unipi.bigdatamining.classification.tree.impurity.FuzzyImpurity]] trait.
  */
private[tree] object FuzzyEntropy extends FuzzyImpurity {
  val log2 : Double = scala.math.log(2)
  private def nlnFunc(x: Double): Double = if (x <= 0) 0D else x*scala.math.log(x)
  
  /**
   * Information calculation for multi-class classification
   * In this class the entropy is used as impurity calculation.
   *
   * @param counts Array[Double] with counts for each label
   * @param totalCount sum of counts for all labels
   * @return information value, or 0 if totalCount = 0
   */
  override def calculate(counts: Array[Double], totalCount: Double): Double = {
    if (totalCount != 0D){
      val impurity = counts.foldLeft(0D)(_ + -nlnFunc(_))
      (impurity + nlnFunc(totalCount)) / (totalCount * log2)
    } else 0D
  }

}

/**
 * Class for updating views of a vector of sufficient statistics,
 * in order to compute impurity from a sample.
 * Note: Instances of this class do not hold the data; they operate on views of the data.
 *
 * @param numClasses  Number of classes for label.
 */
private[tree] class FuzzyEntropyAggregator(numClasses: Int)
  extends FuzzyImpurityAggregator(numClasses) with Serializable {

  /**
    * Update stats for one (node, feature, bin) with the given label.
    *
    * @param allStats  Flat stats array, with stats for this (node, feature, bin) contiguous.
    * @param offset    Start index of stats for this (node, feature, bin).
    */
  def update(allStats: Array[Double], offset: Int, label: Double, value: Double): Unit = {
    if (label >= statsSize) {
      throw new IllegalArgumentException(s"EntropyAggregator given label $label" +
        s" but requires label < numClasses (= $statsSize).")
    }
    if (label < 0) {
      throw new IllegalArgumentException(s"EntropyAggregator given label $label" +
        s"but requires label is non-negative.")
    }
    allStats(offset + label.toInt) += value
  }

  /**
    * Get an [[FuzzyEntropyCalculator]] for a (node, feature, bin).
    *
    * @param allStats  Flat stats array, with stats for this (node, feature, bin) contiguous.
    * @param offset    Start index of stats for this (node, feature, bin).
    */
  def getCalculator(allStats: Array[Double], offset: Int): FuzzyEntropyCalculator = {
    new FuzzyEntropyCalculator(allStats.view(offset, offset + statsSize).toArray)
  }

}

/**
  * Stores statistics for one (node, feature, bin) for calculating impurity.
  * Unlike [[FuzzyEntropyAggregator]], this class stores its own data and is for a specific
  * (node, feature, bin).
  *
  * @param stats  Array of sufficient statistics for a (node, feature, bin).
  */
private[tree] class FuzzyEntropyCalculator(stats: Array[Double]) extends FuzzyImpurityCalculator(stats) {

  /**
    * Make a deep copy of this [[FuzzyEntropyCalculator]].
    */
  def copy: FuzzyEntropyCalculator = new FuzzyEntropyCalculator(stats.clone())

  /**
    * Calculate the impurity from the stored sufficient statistics.
    */
  def calculate(): Double = FuzzyEntropy.calculate(stats, stats.sum)

  /**
    * Number of data points accounted for in the sufficient statistics.
    */
  def count: Long = stats.sum.toLong
  
  /**
    * Prediction which should be made based on the sufficient statistics.
    */
  def predict: Double = if (count == 0) {
    0
  } else {
    indexOfLargestArrayElement(stats)
  }

  /**
    * Return a string with a summary of the Fuzzy Entropy calculator.
    */
  override def toString: String = s"FuzzyEntropyCalculator(stats = [${stats.mkString(", ")}])"

}