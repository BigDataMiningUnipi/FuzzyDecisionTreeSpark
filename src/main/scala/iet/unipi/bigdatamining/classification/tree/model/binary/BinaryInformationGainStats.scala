package iet.unipi.bigdatamining.classification.tree.model.binary

import iet.unipi.bigdatamining.classification.tree.impurity.FuzzyImpurityCalculator
import iet.unipi.bigdatamining.classification.tree.model.PredictStats

/**
  * Information gain statistics for each split
  *
  * @param gain information gain value
  * @param impurity current node impurity
  * @param leftImpurity left child node impurity
  * @param rightImpurity right child node impurity
  * @param leftPredict left child node prediction
  * @param rightPredict right child node prediction
  */
class BinaryInformationGainStats(
                                  val gain: Double,
                                  val impurity: Double,
                                  val leftImpurity: Double,
                                  val rightImpurity: Double,
                                  val leftPredict: PredictStats,
                                  val rightPredict: PredictStats) extends Serializable {

  /**
    * Return a string with a summary of the Binary information gain stats.
    */
  override def toString: String = {
    s"gain = $gain, impurity = $impurity, left impurity = $leftImpurity, " +
      s"right impurity = $rightImpurity"
  }

  override def equals(o: Any): Boolean = o match {
    case other: BinaryInformationGainStats =>
      gain == other.gain &&
        impurity == other.impurity &&
        leftImpurity == other.leftImpurity &&
        rightImpurity == other.rightImpurity &&
        leftPredict == other.leftPredict &&
        rightPredict == other.rightPredict

    case _ => false
  }

  override def hashCode: Int = {
    com.google.common.base.Objects.hashCode(
      gain: java.lang.Double,
      impurity: java.lang.Double,
      leftImpurity: java.lang.Double,
      rightImpurity: java.lang.Double,
      leftPredict,
      rightPredict)
  }
}

/**
  * Companion object of BinaryInformationGainStats
  */
object BinaryInformationGainStats {

  /**
    * An invalid instance that stores the information gain statistics for a multi node.
    * Indeed, the current split doesn't satisfies minimum info gain
    * or minimum number of instances per node.
    *
    * @return An [[iet.unipi.bigdatamining.classification.tree.model.binary.BinaryInformationGainStats]] instance
    * to denote that current split doesn't satisfies minimum info gain
    * or minimum number of instances per node.
    */
  val invalidInformationGainStats = new BinaryInformationGainStats(Double.MinValue,
    -1D, -1D, -1D, PredictStats.empty, PredictStats.empty)

}

/**
  * Fuzzy Impurity statistics for each split
  *
  * @param gain information gain value
  * @param fuzzyImpurity current node impurity
  * @param fuzzyImpurityCalculator fuzzy impurity statistics for current node
  * @param leftChildFuzzyImpurityCalculator fuzzy impurity statistics for left child node
  * @param rightChildFuzzyImpurityCalculator fuzzy impurity statistics for right child node
  * @param valid whether the current split satisfy minimum info gain or
  *              minimum number of instances per node
  */
private[tree] class FuzzyImpurityStats(
                                        val gain: Double,
                                        val fuzzyImpurity: Double,
                                        val fuzzyImpurityCalculator: FuzzyImpurityCalculator,
                                        val leftChildFuzzyImpurityCalculator: FuzzyImpurityCalculator,
                                        val rightChildFuzzyImpurityCalculator: FuzzyImpurityCalculator,
                                        val valid: Boolean = true) extends Serializable {

  /**
    * Return a string with a summary of the Fuzzy Impurity Stats.
    */
  override def toString: String = {
    s"gain = $gain, impurity = $fuzzyImpurity, left impurity = $leftChildFuzzyImpurity " +
      s"right impurity = $rightChildFuzzyImpurity"
  }

  /**
    * Get left child node impurity
    *
    * @return the fuzzy impurity of the left child node
    */
  def leftChildFuzzyImpurity = if (leftChildFuzzyImpurityCalculator != null) {
    leftChildFuzzyImpurityCalculator.calculate()
  } else {
    -1D
  }

  /**
    * Get right child node impurity
    *
    * @return the fuzzy impurity of the right child node
    */
  def rightChildFuzzyImpurity = if (rightChildFuzzyImpurityCalculator != null) {
    rightChildFuzzyImpurityCalculator.calculate()
  } else {
    -1D
  }

}

/**
  * Companion object of FuzzyImpurityStats
  */
private[tree] object FuzzyImpurityStats {

  /**
    * Get an invalid instance that stores the fuzzy impurity statistics.
    * Indeed, the current split doesn't satisfies minimum info gain
    * or minimum number of instances per node.
    *
    * @param fuzzyImpurityCalculator that must be used
    * @return an empty [[iet.unipi.bigdatamining.classification.tree.model.binary.FuzzyImpurityStats]] instance
    */
  def getInvalidImpurityStats(fuzzyImpurityCalculator: FuzzyImpurityCalculator): FuzzyImpurityStats = {
    new FuzzyImpurityStats(Double.MinValue, fuzzyImpurityCalculator.calculate(),
      fuzzyImpurityCalculator, null, null, false)
  }

  /**
    * Get an instance that stores the fuzzy impurity statistics.
    * Indeed, only 'fuzzyImpurity' and 'fuzzyImpurityCalculator' are defined.
    *
    * @param fuzzyImpurityCalculator that must be used
    * @return an [[iet.unipi.bigdatamining.classification.tree.model.binary.FuzzyImpurityStats]] object to
    * that only 'fuzzyImpurity' and 'fuzzyImpurityCalculator' are defined.
    */
  def getEmptyImpurityStats(fuzzyImpurityCalculator: FuzzyImpurityCalculator): FuzzyImpurityStats = {
    new FuzzyImpurityStats(Double.NaN, fuzzyImpurityCalculator.calculate(), fuzzyImpurityCalculator, null, null)
  }

}
