import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Random;

public class LinearPerceptronEnsemble extends AbstractClassifier
{
  LinearPerceptronEnsemble()
  {
    classifiers = new ArrayList<>();
    ensembleSize = 50;
    proportion = 0.5;
  }

  @Override
  public void buildClassifier(Instances instances) throws Exception
  {
    Random r = new Random();
    for (int i = 0; i < ensembleSize; ++i)
    {
      Instances batch = new Instances(instances, 0);
      EnhancedLinearPerceptron elp = new EnhancedLinearPerceptron();
      for (Instance inst : instances)
      {
        if (r.nextDouble() > proportion)
          batch.add(inst);
      }
      classifiers.add(elp);
    }
  }

  private ArrayList<EnhancedLinearPerceptron> classifiers;
  private int ensembleSize;
  private double proportion;
}
