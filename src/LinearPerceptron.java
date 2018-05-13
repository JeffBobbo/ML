import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Random;

public class LinearPerceptron extends AbstractClassifier
{
  public LinearPerceptron()
  {
    maxIterations = 25;
    weights = null;
    bias = 0.0;
  }

  @Override
  public void buildClassifier(Instances instances) throws Exception
  {
    weights = new double[instances.numAttributes()];
    for (int i = 0; i < weights.length; ++i)
    {
      if (i == instances.classIndex())
        weights[i] = 0.0;
      else
        weights[i] = randomWeight();
    }
    boolean fitted = false;
    int it = 0;
    while (!fitted && it++ < maxIterations)
    {
      fitted = true;
      for (Instance instance : instances)
      {
        double yi = classifyInstance(instance);
        if ((yi >= 0.0 && instance.classValue() < 0.0) || (yi < 0.0 && instance.classValue() >= 0.0))
        {
          fitted = false;
          for (int i = 0; i < instance.numAttributes(); ++i)
          {
            if (i == instance.classIndex())
              continue;
            weights[i] += 0.5 * LEARNING_RATE * (instance.classValue() - (yi >= 0.0 ? 1.0 : -1.0)) * instance.value(i) + bias;
          }
        }
      }
    }
  }

  @Override
  public double classifyInstance(Instance instance)
  {
    double y = 0.0;
    for (int i = 0; i < instance.numAttributes(); ++i)
    {
      if (i == instance.classIndex())
        continue;
      y += weights[i] * instance.value(i);
    }
    return y >= 0.0 ? 1.0 : -1.0;
  }

  protected double randomWeight()
  {
    return 1.0;
  }

  public double[] getWeights() { return weights; }
  public void setBias(double b) { bias = b; }
  public double getBias() { return bias; }


  protected final double LEARNING_RATE = 1.0;
  protected int maxIterations;
  protected double[] weights;
  protected double bias;
}
