import weka.core.Instances;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

/**
 * Created by james on 29/04/18.
 */
public class Arsenal
{
  public static Instances loadData(String source) throws IOException
  {
    FileReader reader = new FileReader(source);
    return new Instances(reader);
  }
  public static void main(String[] args)
  {
    String trainingData = "Arsenal_TRAIN.arff";
    Instances train, test;
    try
    {
      train = loadData(trainingData);
      test = loadData("Arsenal_TEST.arff");
    }
    catch (IOException e)
    {
      System.out.println(e.getMessage());
      return;
    }

    System.out.println(train.numInstances());
    System.out.println(test.numInstances());

    for (double a : test.instance(4).toDoubleArray())
      System.out.println(a);
  }
}
