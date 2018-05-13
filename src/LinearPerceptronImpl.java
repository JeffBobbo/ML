import weka.core.Instance;
import weka.core.Instances;

import java.io.FileReader;
import java.io.IOException;

public class LinearPerceptronImpl
{
  private static Instances loadData(String source) throws IOException
  {
    FileReader reader = new FileReader(source);
    return new Instances(reader);
  }

  private static void experimentStandardization(String file, Instances instances, double trainFrac) throws Exception
  {
    int trainNum = (int)(instances.size() * trainFrac);
    Instances train = new Instances(instances, 0, trainNum);
    Instances test = new Instances(instances, trainNum, instances.size() - trainNum);

    EnhancedLinearPerceptron c0 = new EnhancedLinearPerceptron();
    EnhancedLinearPerceptron c1 = new EnhancedLinearPerceptron();
    c1.useStandardization(false);

    c0.buildClassifier(train);
    c1.buildClassifier(train);

    int c0Right = 0, c1Right = 0;
    for (Instance instance : test)
    {
      double r0 = c0.classifyInstance(instance);
      double r1 = c1.classifyInstance(instance);
      double ex = instance.classValue();

      if (ex >= 0.0 && r0 >= 0.0 || ex < 0.0 && r0 < 0.0)
        ++c0Right;
      if (ex >= 0.0 && r1 >= 0.0 || ex < 0.0 && r1 < 0.0)
        ++c1Right;
    }
    System.out.println(file + "," + instances.size() + "," + train.size() + "," + test.size() + "," + instances.numAttributes() + "," + c0Right + "," + c1Right);
  }

  public static void main(String[] args)
  {
    String dataFile = args[0];
    Instances data;
    try
    {
      data = loadData(dataFile);
    }
    catch (IOException e)
    {
      System.err.println(e.getMessage());
      return;
    }

    data.setClassIndex(data.numAttributes() - 1);

    try
    {
      experimentStandardization(dataFile, data, 0.5);
    }
    catch (Exception e)
    {
      System.err.println(e.getMessage());
    }
  }
}
