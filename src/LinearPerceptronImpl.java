import util.Stopwatch;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instance;
import weka.core.Instances;

import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

public class LinearPerceptronImpl
{
  private static Instances loadData(String source) throws IOException
  {
    FileReader reader = new FileReader(source);
    return new Instances(reader);
  }

  private static Instances[] createTrainTestSet(Instances instances, double split) throws Exception
  {
    Instances[] sets = new Instances[2];

    int num = (int)(instances.size() * split);
    instances.randomize(new Random(0));
    sets[0] = new Instances(instances, 0, num);
    sets[1] = new Instances(instances, num, instances.size()-num);

    return sets;
  }

  private static void experimentStandardization(String file, Instances instances, double trainFrac) throws Exception
  {
    Instances[] data = createTrainTestSet(instances, trainFrac);
    Instances train = data[0];
    Instances test = data[1];
    
    EnhancedLinearPerceptron c0 = new EnhancedLinearPerceptron();
    EnhancedLinearPerceptron c1 = new EnhancedLinearPerceptron();
    c1.useStandardization(false);

    Stopwatch c0TimeBuild = new Stopwatch();
    Stopwatch c1TimeBuild = new Stopwatch();

    c0TimeBuild.start();
    c0.buildClassifier(train);
    c0TimeBuild.end();
    c1TimeBuild.start();
    c1.buildClassifier(train);
    c1TimeBuild.end();

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
    System.out.println(file + "," + instances.size() + "," + train.size() + "," + test.size() + "," + instances.numAttributes() + "," + c0Right + "," + c1Right + "," + c0TimeBuild.getElapsedTime() + "," + c1TimeBuild.getElapsedTime());
  }

  private static void experiementCross(String file, Instances instances, double trainFrac) throws Exception
  {
    Instances[] data = createTrainTestSet(instances, trainFrac);
    Instances train = data[0];
    Instances test = data[1];

    EnhancedLinearPerceptron c0 = new EnhancedLinearPerceptron();
    EnhancedLinearPerceptron c1 = new EnhancedLinearPerceptron();
    c1.useModelSelection(true);

    Stopwatch c0TimeBuild = new Stopwatch();
    Stopwatch c1TimeBuild = new Stopwatch();

    c0TimeBuild.start();
    c0.buildClassifier(train);
    c0TimeBuild.end();
    c1TimeBuild.start();
    c1.buildClassifier(train);
    c1TimeBuild.end();

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
    System.out.println(file + "," + instances.size() + "," + train.size() + "," + test.size() + "," + instances.numAttributes() + "," + c0Right + "," + c1Right + "," + c0TimeBuild.getElapsedTime() + "," + c1TimeBuild.getElapsedTime());
  }

  private static void experiementCompare(String file, Instances instances, double trainFrac) throws Exception
  {
    Instances[] data = createTrainTestSet(instances, trainFrac);
    Instances train = data[0];
    Instances test = data[1];

    EnhancedLinearPerceptron c0 = new EnhancedLinearPerceptron();
    LinearPerceptronEnsemble c1 = new LinearPerceptronEnsemble();
    MultilayerPerceptron c2 = new MultilayerPerceptron();

    Stopwatch c0Timer = new Stopwatch();
    Stopwatch c1Timer = new Stopwatch();
    Stopwatch c2Timer = new Stopwatch();

    c0Timer.start();
    c0.buildClassifier(test);
    c0Timer.end();
    c1Timer.start();
    c1.buildClassifier(test);
    c1Timer.end();
    c2Timer.start();
    c2.buildClassifier(test);
    c2Timer.end();

    int c0Right = 0, c1Right = 0, c2Right = 0;
    for (Instance instance : test)
    {
      double r0 = c0.classifyInstance(instance);
      double r1 = c1.classifyInstance(instance);
      double r2 = c2.classifyInstance(instance);
      double ex = instance.classValue();

      if (ex >= 0.0 && r0 >= 0.0 || ex < 0.0 && r0 < 0.0)
        ++c0Right;
      if (ex >= 0.0 && r1 >= 0.0 || ex < 0.0 && r1 < 0.0)
        ++c1Right;
      if (ex >= 0.0 && r2 >= 0.0 || ex < 0.0 && r2 < 0.0)
        ++c2Right;
    }
    System.out.println(file + "," + instances.size() + "," + train.size() + "," + test.size() + "," + instances.numAttributes() + "," + c0Right + "," + c1Right + "," + c2Right + "," + c0Timer.getElapsedTime() + "," + c1Timer.getElapsedTime() + "," + c2Timer.getElapsedTime());
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
      for (int d = 1; d < 10; ++d)
        experimentStandardization("standard", data, d / 10.0);
      for (int d = 1; d < 10; ++d)
        experiementCross("cross", data, d / 10.0);
      for (int d = 1; d < 10; ++d)
        experiementCompare("compare", data, d/ 10.0);
    }
    catch (Exception e)
    {
      System.err.println(e.getMessage());
    }
  }
}
