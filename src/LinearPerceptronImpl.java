import weka.core.Instances;

import java.io.FileReader;
import java.io.IOException;

/**
 * Created by james on 29/04/18.
 */
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
      data= loadData("PerceptronTEST.arff");
    }
    catch (IOException e)
    {
      System.out.println(e.getMessage());
      return;
    }

    System.out.println(data.numInstances());
    LinearPerceptron lp = new LinearPerceptron();

    try
    {
      lp.buildClassifier(data);
    } catch (Exception e)
    {
      e.printStackTrace();
    }
  }
}
