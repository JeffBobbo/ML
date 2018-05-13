import weka.classifiers.AbstractClassifier;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.RandomSubset;

import java.util.ArrayList;

public class LinearPerceptronEnsemble extends AbstractClassifier
{
  LinearPerceptronEnsemble()
  {
    classifiers = new ArrayList<>();
    filters = new ArrayList<>();
    ensembleSize = 50;
    proportion = 0.5;
  }

  @Override
  public void buildClassifier(Instances instances) throws Exception
  {
    for (int i = 0; i < ensembleSize; ++i)
    {
      RandomSubset f = new RandomSubset();
      f.setNumAttributes(proportion);
      f.setInputFormat(instances);
      for (Instance inst : instances)
        f.input(inst);
      f.batchFinished();
      Instances block = f.getOutputFormat();
      Instance processed;
      while ((processed = f.output()) != null)
        block.add(processed);
      EnhancedLinearPerceptron elp = new EnhancedLinearPerceptron();
      classifiers.add(elp);
      filters.add(f);
      elp.buildClassifier(block);
    }
  }

  @Override
  public double classifyInstance(Instance instance) throws Exception
  {
    double y = 0.0;
    for (int i = 0; i < ensembleSize; ++i)
    {
      filters.get(i).input(instance);
      filters.get(i).batchFinished();
      Instance inst = filters.get(i).output();
      y += classifiers.get(i).classifyInstance(inst);
    }

    return y >= 0.0 ? 1.0 : -1.0;
  }

  @Override
  public double[] distributionForInstance(Instance instance) throws Exception
  {
    double[] r = new double[2];
    r[0] = r[1] = 0.0;

    for (int i = 0; i < ensembleSize; ++i)
    {
      filters.get(i).input(instance);
      filters.get(i).batchFinished();
      double c = classifiers.get(i).classifyInstance(filters.get(i).output());
      ++r[c >= 0 ? 1 : 0];
    }
    r[0] /= ensembleSize;
    r[1] /= ensembleSize;
    return r;
  }

  private ArrayList<EnhancedLinearPerceptron> classifiers;
  private ArrayList<RandomSubset> filters;
  private int ensembleSize;
  private double proportion;
}
