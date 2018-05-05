import weka.core.Instance;
import weka.core.Instances;

public class EnhancedLinearPerceptron extends LinearPerceptron
{
  public enum PerceptronModel
  {
    OFFLINE,
    ONLINE
  }

  EnhancedLinearPerceptron()
  {
    super();
    standardise = true;
    model = PerceptronModel.ONLINE;
  }

  EnhancedLinearPerceptron(boolean std, PerceptronModel m)
  {
    super();
    standardise = std;
    model = m;
  }

  double mean(Instances instances, final int i)
  {
    double sum = 0.0;
    for (Instance inst : instances)
      sum += inst.value(i);
    return sum / instances.numInstances();
  }

  double stddev(Instances instances, final int i)
  {
    double v = 0.0;
    final double m = mean(instances, i);

    for (Instance inst : instances)
    {
      v += Math.pow(inst.value(i) - m, 2.0);
    }

    return Math.sqrt(v / instances.numInstances());
  }

  private void buildOffline(Instances instances)
  {
    weightX = 0.0;
    weightY = 0.0;

    boolean fitted = false;
    while (!fitted)
    {
      fitted = true;
      double dWeightX = 0.0, dWeightY = 0.0;
      for (Instance inst : instances)
      {
        double yi = psi(inst);
        if ((yi >= 0.0 && inst.value(2) < 0.0) || (yi < 0.0 && inst.value(2) >= 0.0))
        {
          fitted = false;
          dWeightX += 0.5 * LEARNING_RATE * (inst.value(2) - (yi >= 0.0 ? 1.0 : -1.0)) * inst.value(0) + bias;
          dWeightY += 0.5 * LEARNING_RATE * (inst.value(2) - (yi >= 0.0 ? 1.0 : -1.0)) * inst.value(1) + bias;
        }
      }
      weightX += dWeightX;
      weightY += dWeightY;
    }
  }

  @Override
  public void buildClassifier(Instances instances) throws Exception
  {
    if (model == PerceptronModel.ONLINE)
      super.buildClassifier(instances);
    else
      buildOffline(instances);
  }

  @Override
  public double classifyInstance(Instance instance)
  {
    double y1 = weightX * instance.value(0) + weightY * instance.value(1);
    return y1 >= 0.0 ? 1.0 : -1.0;
  }

  public double getWeightX() { return weightX; }
  public double getWeightY() { return weightY; }

  boolean standardise;
  PerceptronModel model;

}
