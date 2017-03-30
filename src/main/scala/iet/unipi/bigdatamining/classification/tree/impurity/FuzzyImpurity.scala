package iet.unipi.bigdatamining.classification.tree.impurity

/**
  * A Fuzzy Impurity trait to define the common method
  * that each fuzzy impurity instance should implement.
  */
private[tree] trait FuzzyImpurity extends Serializable {

  /**
   * Information calculation for multiclass classification
   * @param counts Array[Double] with counts for each label
   * @param totalCount sum of counts for all labels
   * @return information value, or 0 if totalCount = 0
   */
  def calculate(counts: Array[Double], totalCount: Double): Double

}

/**
 * Interface for updating views of a vector of sufficient statistics,
 * in order to compute impurity from a sample.
 * Note: Instances of this class do not hold the data; they operate on views of the data.
 * @param statsSize  Length of the vector of sufficient statistics for one bin.
 */
private[tree] abstract class FuzzyImpurityAggregator(val statsSize: Int) extends Serializable {

  /**
   * Merge the stats from one bin into another.
   * @param allStats  Flat stats array, with stats for this (node, feature, bin) contiguous.
   * @param offset    Start index of stats for (node, feature, bin) which is modified by the merge.
   * @param otherOffset  Start index of stats for (node, feature, other bin) which is not modified.
   */
  def merge(allStats: Array[Double], offset: Int, otherOffset: Int): Unit = {
    (0 until statsSize).foreach(i => allStats(offset + i) += allStats(otherOffset + i))
  }

  /**
   * Update stats for one (node, feature, bin) with the given label.
   * @param allStats  Flat stats array, with stats for this (node, feature, bin) contiguous.
   * @param offset    Start index of stats for this (node, feature, bin).
   */
  def update(allStats: Array[Double], offset: Int, label: Double, value: Double): Unit

  /**
   * Get an [[FuzzyImpurityCalculator]] for a (node, feature, bin).
   * @param allStats  Flat stats array, with stats for this (node, feature, bin) contiguous.
   * @param offset    Start index of stats for this (node, feature, bin).
   */
  def getCalculator(allStats: Array[Double], offset: Int): FuzzyImpurityCalculator

}

/**
 * Stores statistics for one (node, feature, bin) for calculating impurity.
 * Unlike [[FuzzyImpurityAggregator]], this class stores its own data and is for a specific
 * (node, feature, bin).
 * @param stats  Array of sufficient statistics for a (node, feature, bin).
 */
private[tree] abstract class FuzzyImpurityCalculator(val stats: Array[Double]) extends Serializable {

  /**
   * Make a deep copy of this [[FuzzyImpurityCalculator]].
   */
  def copy: FuzzyImpurityCalculator

  /**
   * Calculate the fuzzy impurity from the stored sufficient statistics.
   */
  def calculate(): Double

  /**
   * Prediction which should be made based on the sufficient statistics.
   */
  def predict: Double
  
  /**
   * Add the stats from another calculator into this one, modifying and returning this calculator.
   */
  def add(other: FuzzyImpurityCalculator): FuzzyImpurityCalculator = {
    require(stats.length == other.stats.length,
      s"Two FuzzyImpurityCalculator instances cannot be added with different counts sizes." +
        s"  Sizes are ${stats.length} and ${other.stats.length}.")
    var i = 0
    val len = other.stats.length
    while (i < len) {
      stats(i) += other.stats(i) 
      i += 1
    }
    this
  }

  /**
   * Subtract the stats from another calculator from this one, modifying and returning this
   * calculator.
   */
  def subtract(other: FuzzyImpurityCalculator): FuzzyImpurityCalculator = {
    require(stats.length == other.stats.length,
      s"Two FuzzyImpurityCalculator instances cannot be subtracted with different counts sizes." +
      s"  Sizes are ${stats.length} and ${other.stats.length}.")
    var i = 0
    val len = other.stats.length
    while (i < len) {
      stats(i) -= other.stats(i)
      i += 1
    }
    this
  }

  /**
   * Number of data points accounted for in the sufficient statistics.
   */
  def count: Long

  /**
   * Return the index of the largest array element.
   * Fails if the array is empty.
   */
  protected def indexOfLargestArrayElement(array: Array[Double]): Int = {
    val result = array.foldLeft(-1, Double.MinValue, 0) {
      case ((maxIndex, maxValue, currentIndex), currentValue) =>
        if (currentValue > maxValue) {
          (currentIndex, currentValue, currentIndex + 1)
        } else {
          (maxIndex, maxValue, currentIndex + 1)
        }
    }
    if (result._1 < 0) {
      throw new RuntimeException("ImpurityCalculator internal error:" +
        " indexOfLargestArrayElement failed")
    }
    result._1
  }
  
}
