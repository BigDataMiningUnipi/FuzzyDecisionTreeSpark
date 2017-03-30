package iet.unipi.bigdatamining.fuzzy.tnorm

/**
  * The object represents the implementation of minimum function tNorm.
  * It implements [[iet.unipi.bigdatamining.fuzzy.tnorm.TNorm]] trait
  */
private[bigdatamining] object MinTNorm extends TNorm{

  @Override
  def calculate(uAi: Double*): Double = {
    uAi.foldLeft(1D)(_ min _)
  }
   
}