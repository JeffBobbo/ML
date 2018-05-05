import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Created by james on 05/05/18.
 */
public class LinearPerceptron extends AbstractClassifier
{
  LinearPerceptron()
  {
    weightX = -1.0;
    weightY =  2.0;
  }

  @Override
  public void buildClassifier(Instances instances) throws Exception
  {
    boolean fitted = false;
    while (fitted == false)
    {
      fitted = true;
      for (Instance instance : instances)
      {
        double y1 = weightX * instance.value(0) + weightY * instance.value(1);
        if ((y1 > 0.0 && instance.value(2) < 0.0) || (y1 < 0.0 && instance.value(2) > 0.0))
        {
          fitted = false;
          weightX = weightX + 0.5 * LEARNING_RATE * (instance.value(2) - (y1 > 0.0 ? 1.0 : -1.0)) * instance.value(0);
          weightY = weightY + 0.5 * LEARNING_RATE * (instance.value(2) - (y1 > 0.0 ? 1.0 : -1.0)) * instance.value(1);
        }
      }
    }
  }

  private final double LEARNING_RATE = 1.0;
  private double weightX;
  private double weightY;
}
