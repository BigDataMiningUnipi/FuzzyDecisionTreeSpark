package iet.unipi.bigdatamining.classification.tree.model.multi

import iet.unipi.bigdatamining.classification.tree.impurity.FuzzyImpurityCalculator
import iet.unipi.bigdatamining.classification.tree.model.PredictStats

/**
  * Information gain statistics for each split
  *
  * @param gain information gain value
  * @param impurity current node impurity
  * @param childrenImpurity children node impurities
  * @param childrenPredict children node predicts
  */
class MultiInformationGainStats(
                                 val gain: Double,
                                 val impurity: Double,
                                 val childrenImpurity: Array[Double],
                                 val childrenPredict: Array[PredictStats]) extends Serializable {

  /**
    * Return a string with a summary of the Multi Information gain stats.
    */
  override def toString: String = {
    "gain = " + gain + " impurity = " + impurity +" childrenImpurity = " + childrenImpurity.mkString("{", ",", "}")
  }

  override def equals(o: Any): Boolean = o match {
    case other: MultiInformationGainStats =>
      gain == other.gain &&
        impurity == other.impurity &&
        (childrenImpurity sameElements other.childrenImpurity) &&
        (childrenPredict sameElements other.childrenPredict)

    case _ => false
  }

  override def hashCode: Int = {
    com.google.common.base.Objects.hashCode(
      gain: java.lang.Double,
      impurity: java.lang.Double,
      java.util.Arrays.asList(childrenImpurity: _*),
      java.util.Arrays.asList(childrenPredict: _*))
  }

}

/**
  * Companion object of MultiInformationGainStats
  */
private[tree] object MultiInformationGainStats {

  /**
    * An invalid instance that stores the information gain statistics for a multi node.
    * Indeed, the current split doesn't satisfies minimum info gain
    * or minimum number of instances per node.
    *
    * @return An [[iet.unipi.bigdatamining.classification.tree.model.multi.MultiInformationGainStats]] instance
    * to denote that current split doesn't satisfies minimum info gain
    * or minimum number of instances per node.
    */
  val invalidInformationGainStats = new MultiInformationGainStats(Double.MinValue, -1D,
    Array.empty[Double], Array.empty[PredictStats])

}

/**
  * Fuzzy Impurity statistics for each split
  *
  * @param gain information gain value
  * @param fuzzyImpurity current fuzzy node impurity
  * @param fuzzyImpurityCalculator fuzzy impurity statistics for current node
  * @param childrenFuzzyImpurityCalculators fuzzy impurity statistics for children nodes
  */
private[tree] class FuzzyImpurityStats(
                                        val gain: Double,
                                        val fuzzyImpurity: Double,
                                        val fuzzyImpurityCalculator: FuzzyImpurityCalculator,
                                        val childrenFuzzyImpurityCalculators: Array[FuzzyImpurityCalculator])
  extends Serializable {

  /**
    * A string that summarizes  of the Fuzzy Impurity Stats.
    *
    * @return a string representation of the Fuzzy Impurity Stats
    */
  override def toString: String = {
    s"gain = $gain, impurity = $fuzzyImpurity, children impurity = $childrenFuzzyImpurityCalculators "
  }

  /**
    * Get the impurity of each child
    *
    * @return an array where each i-th element stores the impurity of i-th child.
    *         Size of the array depends on the number of children
    */
  def childrenImpurity: Array[Double] = if (!childrenFuzzyImpurityCalculators.isEmpty) {
    val childrenFuzzyImpurity = new Array[Double](childrenFuzzyImpurityCalculators.length)
    childrenFuzzyImpurityCalculators.indices.foreach{ i =>
      childrenFuzzyImpurity(i) = childrenFuzzyImpurityCalculators(i).calculate()
    }
    childrenFuzzyImpurity
  } else {
    Array.empty[Double]
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
    * @return an empty [[iet.unipi.bigdatamining.classification.tree.model.multi.FuzzyImpurityStats]] instance
    */
  def getInvalidImpurityStats(fuzzyImpurityCalculator: FuzzyImpurityCalculator): FuzzyImpurityStats = {
    new FuzzyImpurityStats(Double.MinValue, fuzzyImpurityCalculator.calculate(),
      fuzzyImpurityCalculator, Array.empty[FuzzyImpurityCalculator])
  }

  /**
    * Get an instance that stores the fuzzy impurity statistics.
    * Indeed, only 'fuzzyImpurity' and 'fuzzyImpurityCalculator' are defined.
    *
    * @param fuzzyImpurityCalculator that must be used
    * @return an [[iet.unipi.bigdatamining.classification.tree.model.multi.FuzzyImpurityStats]] instance
    * where only 'fuzzyImpurity' and 'fuzzyImpurityCalculator' are defined.
    */
  def getEmptyImpurityStats(fuzzyImpurityCalculator: FuzzyImpurityCalculator): FuzzyImpurityStats = {
    new FuzzyImpurityStats(Double.NaN, fuzzyImpurityCalculator.calculate(), fuzzyImpurityCalculator, null)
  }

}
