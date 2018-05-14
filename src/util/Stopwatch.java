package util;

import java.io.UncheckedIOException;

public class Stopwatch
{
  public Stopwatch()
  {
    startPoint = -1;
    endPoint = -1;
    pausePoint = -1;
  }

  public void start()
  {
    startPoint = System.currentTimeMillis();
  }

  public void end()
  {
    endPoint = System.currentTimeMillis();
  }

  public void pause()
  {
    pausePoint = System.currentTimeMillis();
  }

  public void resume()
  {
    startPoint += (System.currentTimeMillis() - pausePoint);
    pausePoint = -1;
  }

  public long getElapsedTime()
  {
    if (endPoint == -1)
      return -1;
    if (pausePoint != -1)
      return pausePoint - startPoint;
    return endPoint - startPoint;
  }

  private long startPoint;
  private long endPoint;
  private long pausePoint;
}
