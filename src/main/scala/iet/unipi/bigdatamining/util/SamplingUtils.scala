package iet.unipi.bigdatamining.util

import scala.collection.mutable
import scala.reflect.ClassTag
import scala.util.Random

private[bigdatamining] object SamplingUtils {

  /**
   * IMPORTANT: Code extracted from [[org.apache.spark.util.random.SamplingUtils]]
   *
   * Reservoir sampling implementation that also returns the input size.
   *
   * @param input input size
   * @param excludingSet the T elem that must not be included in
   *                     random selection
   * @param k reservoir size
   * @param seed random seed
   * @return (samples, input size)
   */
  def reservoirSampleAndCount[T: ClassTag](
      input: Iterator[T],
      excludingSet: Array[T],
      k: Int,
      seed: Long = Random.nextLong()): (Array[T], Int) = {
    val reservoirBuffer = mutable.ArrayBuffer.empty[T]
    // Put the first k elements in the reservoir.
    var i = 0
    while (reservoirBuffer.length < k && input.hasNext) {
      val item = input.next()
      if (!excludingSet.contains(item))
        reservoirBuffer += item
      i += 1
    }

    // If we have consumed all the elements, return them. Otherwise do the replacement.
    if (reservoirBuffer.length < k) {
      (reservoirBuffer.toArray, i)
    } else {
      // If input size > k, continue the sampling process.
      val rand = new XORShiftRandom(seed)
      val reservoir = reservoirBuffer.toArray
      while (input.hasNext) {
        val item = input.next()
        if (!excludingSet.contains(item)){
          val replacementIndex = rand.nextInt(i)
          if (replacementIndex < k) {
            reservoir(replacementIndex) = item
          }
        }
        i += 1
      }
      (reservoir, i)
    }
  }
  
}