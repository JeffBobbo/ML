import weka.core.Instances;

import java.io.FileReader;
import java.io.IOException;

public class LinearPerceptronImpl
{
  public static Instances loadData(String source) throws IOException
  {
    FileReader reader = new FileReader(source);
    return new Instances(reader);
  }
  public static void main(String[] args)
  {
    Instances data;
    try
    {
      data = loadData("data/balloons.arff");
      //data = loadData("data/Perceptron.arff");
    }
    catch (IOException e)
    {
      System.out.println(e.getMessage());
      return;
    }

    data.setClassIndex(data.numAttributes() - 1);
    //LinearPerceptron lp = new LinearPerceptron();
    LinearPerceptron lp = new EnhancedLinearPerceptron(true, EnhancedLinearPerceptron.PerceptronModel.ONLINE, false);
    //((EnhancedLinearPerceptron)lp).useModelSelection(true);

    try
    {
      lp.buildClassifier(data);
    }
    catch (Exception e)
    {
      e.printStackTrace();
      return;
    }

    double[] a = lp.getWeights();
    for (double d : a)
      System.out.println(d);
  }
}
