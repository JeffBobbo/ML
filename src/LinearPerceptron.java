import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

public class LinearPerceptron extends AbstractClassifier
{
  public LinearPerceptron()
  {
    weightX = 1.0;
    weightY = 1.0;
    bias = 0.0;
  }
  public LinearPerceptron(double initWeightX, double initWeightY, double b)
  {
    weightX = initWeightX;
    weightY = initWeightY;
    bias = b;
  }

  protected double psi(Instance instance)
  {
    return weightX * instance.value(0) + weightY * instance.value(1);
  }

  @Override
  public void buildClassifier(Instances instances) throws Exception
  {
    boolean fitted = false;
    while (!fitted)
    {
      fitted = true;
      for (Instance instance : instances)
      {
        double yi = psi(instance);
        if ((yi >= 0.0 && instance.value(2) < 0.0) || (yi < 0.0 && instance.value(2) >= 0.0))
        {
          fitted = false;
          weightX = weightX + 0.5 * LEARNING_RATE * (instance.value(2) - (yi >= 0.0 ? 1.0 : -1.0)) * instance.value(0) + bias;
          weightY = weightY + 0.5 * LEARNING_RATE * (instance.value(2) - (yi >= 0.0 ? 1.0 : -1.0)) * instance.value(1) + bias;
        }
      }
    }
  }

  @Override
  public double classifyInstance(Instance instance)
  {
    double y1 = weightX * instance.value(0) + weightY * instance.value(1);
    return y1 >= 0.0 ? 1.0 : -1.0;
  }

  public double getWeightX() { return weightX; }
  public double getWeightY() { return weightY; }

  protected final double LEARNING_RATE = 1.0;
  protected final double bias;
  protected double weightX;
  protected double weightY;
}
