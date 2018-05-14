import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Random;

public class EnhancedLinearPerceptron extends LinearPerceptron
{
  public enum PerceptronModel
  {
    OFFLINE,
    ONLINE
  }

  public EnhancedLinearPerceptron()
  {
    super();
    standardise = true;
    mean = null;
    variance = null;
    count = 0;
    model = PerceptronModel.ONLINE;
    modelSel = false;
  }

  private double mean(final Instances instances, final int attr)
  {
    return instances.meanOrMode(attr);
  }

  private void standardise(Instances instances)
  {
    count = instances.numInstances();
    mean = new double[instances.numAttributes()];
    for (int i = 0; i < mean.length; ++i)
    {
      if (i != instances.classIndex())
        mean[i] = mean(instances, i);
    }
    variance = new double[instances.numAttributes()];
    for (int i = 0; i < variance.length; ++i)
    {
      if (i != instances.classIndex())
        variance[i] = instances.variance(i);
    }

    for (Instance inst : instances)
    {
      for (int i = 0; i < inst.numAttributes(); ++i)
      {
        if (i != inst.classIndex())
          inst.setValue(i, (inst.value(i) - mean[i]) / Math.sqrt(variance[i]));
      }
    }
  }

  private void buildOffline(Instances instances)
  {
    if (standardise)
      standardise(instances);
    weights = new double[instances.numAttributes()];
    for (int i = 0; i < weights.length; ++i)
      weights[i] = randomWeight();

    boolean fitted = false;
    int it = 0;
    while (!fitted && it++ < maxIterations)
    {
      fitted = true;
      double dWeights[] = new double[weights.length];
      for (Instance inst : instances)
      {
        double yi = super.classifyInstance(inst);
        if ((yi >= 0.0 && inst.classValue() < 0.0) || (yi < 0.0 && inst.classValue() >= 0.0))
        {
          fitted = false;
          for (int i = 0; i < instances.numAttributes()-1; ++i)
          {
            if (i == instances.classIndex())
              continue;
            dWeights[i] += 0.5 * LEARNING_RATE * (inst.classValue() - (yi >= 0.0 ? 1.0 : -1.0)) * inst.value(i) + bias;
          }
        }
      }
      for (int i = 0; i < weights.length; ++i)
        weights[i] += dWeights[i];
    }
  }

  @Override
  public void buildClassifier(Instances instances) throws Exception
  {
    if (standardise)
      standardise(instances);

    PerceptronModel pm = modelSel ? modelSelection(instances) : model;
    if (pm == PerceptronModel.ONLINE)
      super.buildClassifier(instances);
    else
      buildOffline(instances);
  }

  @Override
  public double classifyInstance(Instance instance)
  {
    double y = 0.0;
    for (int i = 0; i < instance.numAttributes(); ++i)
    {
      if (i == instance.classIndex())
        continue;

      double v = instance.value(i);

      if (standardise)
      {
        double sum = mean[i] * count;
        double dev = variance[i] * count;

        sum += v;
        double mean = sum / (count + 1);
        dev += Math.pow(v - mean, 2.0);
        double stddev = dev / (count + 1);
        v = (v - mean) / stddev;
      }
      y += weights[i] * v;
    }
    return y >= 0.0 ? 1.0 : -1.0;
  }

  private PerceptronModel modelSelection(Instances instances) throws Exception
  {
    final int count = instances.size();
    final int groupSize = Math.max(1, count / 4);

    int onlineVotes = 0, offlineVotes = 0;
    for (int i = 0; i < instances.size(); i += groupSize)
    {
      int tStart0 = 0, tEnd0 = Math.min(count,i), tStart1 = Math.min(count, i+groupSize);
      Instances test = new Instances(instances, 0);
      if (tEnd0 != 0)
        test.addAll(new Instances(instances, tStart0, tEnd0 - tStart0));
      if (count > tStart1)
        test.addAll(new Instances(instances, tStart1, count - tStart1));
      Instances train = new Instances(instances, tEnd0, tStart1 - tEnd0);

      int onlineCorrect = 0, offlineCorrect = 0;

      EnhancedLinearPerceptron elp = new EnhancedLinearPerceptron();
      elp.buildClassifier(train);
      for (Instance inst : test)
      {
        final double r = elp.classifyInstance(inst);
        if (r >= 0.0 && inst.classValue() >= 0.0 || r < 0.0 && inst.classValue() < 0.0)
          ++onlineCorrect;
      }
      elp = new EnhancedLinearPerceptron();
      elp.useStandardization(standardise);
      elp.useModel(PerceptronModel.OFFLINE);
      elp.buildClassifier(train);
      for (Instance inst : test)
      {
        final double r = elp.classifyInstance(inst);
        if (r >= 0 && inst.classValue() >= 0.0 || r < 0.0 && inst.classValue() < 0.0)
          ++offlineCorrect;
      }

      if (onlineCorrect > offlineCorrect)
        ++onlineVotes;
      else
        ++offlineVotes;
    }
    return onlineVotes > offlineVotes ? PerceptronModel.ONLINE : PerceptronModel.OFFLINE;
  }

  public void useStandardization(boolean s) { standardise = s; }
  public void useModel(PerceptronModel m) { model = m; }
  public void useModelSelection(boolean b) { modelSel = b; }

  private double[] mean;
  private double[] variance;
  private int count;

  // standardize inputs (0 mean and 1 std dev), defaults to true
  private boolean standardise;
  // use online or offline, defaults to online
  private PerceptronModel model;
  // determine whether to use online or offline based on which is more accurate, defaults to false
  private boolean modelSel;
}
