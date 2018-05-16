#!/usr/bin/perl

use warnings;
use strict;

use Data::Dumper;

my $results = {};

opendir(my $dh, 'results/') or die "$!\n";
my @files = grep(/\.csv/, readdir($dh));
closedir($dh);

foreach (@files)
{
  open(my $fh, '<', 'results/' . $_) or die "$!\n";
  chomp(my @lines = <$fh>);
  close($fh);
  my $i = 0;
  foreach my $line (@lines)
  {
    ++$i;
    $i = 1 if ($i == 10);
    my @data = split(',', $line);
    my $test = shift(@data);
    $results->{$test}{$_}{instances} = shift(@data);
    $results->{$test}{$_}{train}{$i} = shift(@data);
    $results->{$test}{$_}{test}{$i} = shift(@data);
    $results->{$test}{$_}{attributes} = shift(@data);
    if ($test eq 'standard')
    {
      my ($c0r, $c1r, $c0t, $c1t) = @data;
      $results->{$test}{$_}{standardized}{$i} = {'right' => $c0r, 'time' => $c0t};
      $results->{$test}{$_}{nonstandardized}{$i} = {'right' => $c1r, 'time' => $c1t};
    }
    elsif ($test eq 'cross')
    {
      my ($c0r, $c1r, $c0t, $c1t) = @data;
      $results->{$test}{$_}{normal}{$i} = {'right' => $c0r, 'time' => $c0t};
      $results->{$test}{$_}{xtrain}{$i} = {'right' => $c1r, 'time' => $c1t};
    }
    elsif ($test eq 'compare')
    {
      my ($c0r, $c1r, $c2r, $c0t, $c1t, $c2t) = @data;
      $results->{$test}{$_}{elp}{$i} = {'right' => $c0r, 'time' => $c0t};
      $results->{$test}{$_}{lpe}{$i} = {'right' => $c1r, 'time' => $c1t};
      $results->{$test}{$_}{ml}{$i}  = {'right' => $c2r, 'time' => $c2t};
    }
  }
}

accStdAverage();
accStdLarge();
accCrossAverage();
accCompareAverage();
runtime();

# now all the data is loaded, produce some graphs
sub accStdAverage
{
  my $data = '';
  # produce the data to plot
  for my $i (1..9)
  {
    my $a0 = 0;
    my $a1 = 0;
    my $n  = 0;
    for my $f (keys(%{$results->{standard}}))
    {
      ++$n;
      my $test = $results->{standard}{$f}{test}{$i};
      my $r0 = $results->{standard}{$f}{standardized}{$i}{right};
      my $r1 = $results->{standard}{$f}{nonstandardized}{$i}{right};
      $a0 += $r0 / $test;
      $a1 += $r1 / $test;
    }
    $data .= $i * 10 . ',' . ($a0 / $n) * 100 . ','. ($a1 / $n) * 100 . "\n";
  }

  open(my $fh, '>', 'tmp.csv') or die "$!\n";
  print $fh $data;
  close($fh);

  my $plot = <<'EOH';
set title 'Accuracy - Standardization comparison'
set ylabel 'Accuracy (%)'
set xlabel 'Train set size (%)'
set grid
set term png
set output 'acc-stand.png'
set datafile separator ","
plot 'tmp.csv' using 1:2 with lines linetype 1 title "With standardization", 'tmp.csv' using 1:3 with lines linetype 2 title "Without standardization"
EOH

  open($fh, '>', 'plot') or die "$!\n";
  print $fh $plot;
  close($fh);

  system("gnuplot plot");
}

sub accStdLarge
{
  my $data = '';
  # produce the data to plot
  for my $i (1..9)
  {
    my $test = $results->{standard}{'spambase.arff.csv'}{test}{$i};
    my $r0 = $results->{standard}{'spambase.arff.csv'}{standardized}{$i}{right};
    my $r1 = $results->{standard}{'spambase.arff.csv'}{nonstandardized}{$i}{right};
    my $accuracy0 = $r0 / $test;
    my $accuracy1 = $r1 / $test;
    $data .= $i * 10 . ',' . $accuracy0 * 100 . ','. $accuracy1 * 100 . "\n";
  }

  open(my $fh, '>', 'tmp.csv') or die "$!\n";
  print $fh $data;
  close($fh);

  my $plot = <<'EOH';
set title 'Accuracy - Standardization comparison'
set ylabel 'Accuracy (%)'
set xlabel 'Train set size (%)'
set grid
set term png
set output 'acc-stand-spambase.png'
set datafile separator ","
plot 'tmp.csv' using 1:2 with lines linetype 1 title "With standardization", 'tmp.csv' using 1:3 with lines linetype 2 title "Without standardization"
EOH

  open($fh, '>', 'plot') or die "$!\n";
  print $fh $plot;
  close($fh);

  system("gnuplot plot");
}

sub accCrossAverage
{
  my $data = '';
  # produce the data to plot
  for my $i (1..9)
  {
    my $a0 = 0;
    my $a1 = 0;
    my $n  = 0;
    for my $f (keys(%{$results->{cross}}))
    {
      ++$n;
      my $test = $results->{cross}{$f}{test}{$i};
      my $r0 = $results->{cross}{$f}{normal}{$i}{right};
      my $r1 = $results->{cross}{$f}{xtrain}{$i}{right};
      $a0 += $r0 / $test;
      $a1 += $r1 / $test;
    }
    $data .= $i * 10 . ',' . ($a0 / $n) * 100 . ','. ($a1 / $n) * 100 . "\n";
  }

  open(my $fh, '>', 'tmp.csv') or die "$!\n";
  print $fh $data;
  close($fh);

  my $plot = <<'EOH';
set title 'Accuracy - Cross-validation training comparison'
set ylabel 'Accuracy (%)'
set xlabel 'Train set size (%)'
set grid
set term png
set output 'acc-cross.png'
set datafile separator ","
plot 'tmp.csv' using 1:2 with lines linetype 1 title "On-line", 'tmp.csv' using 1:3 with lines linetype 2 title "With cross-validation training"
EOH

  open($fh, '>', 'plot') or die "$!\n";
  print $fh $plot;
  close($fh);

  system("gnuplot plot");
}

sub accCompareAverage
{
  my $data = '';
  # produce the data to plot
  for my $i (1..9)
  {
    my $a0 = 0;
    my $a1 = 0;
    my $a2 = 0;
    my $n  = 0;
    for my $f (keys(%{$results->{compare}}))
    {
      ++$n;
      my $test = $results->{compare}{$f}{test}{$i};
      my $r0 = $results->{compare}{$f}{elp}{$i}{right};
      my $r1 = $results->{compare}{$f}{lpe}{$i}{right};
      my $r2 = $results->{compare}{$f}{ml}{$i}{right};
      $a0 += $r0 / $test;
      $a1 += $r1 / $test;
      $a2 += $r2 / $test;
    }
    $data .= $i * 10 . ',' . ($a0 / $n) * 100 . ','. ($a1 / $n) * 100 . ','. ($a2 / $n) * 100 . "\n";
  }

  open(my $fh, '>', 'tmp.csv') or die "$!\n";
  print $fh $data;
  close($fh);

  my $plot = <<'EOH';
set title 'Accuracy - Comparison of classifiers'
set ylabel 'Accuracy (%)'
set xlabel 'Train set size (%)'
set grid
set term png
set output 'acc-compare.png'
set datafile separator ","
plot 'tmp.csv' using 1:2 with lines linetype 1 title "EnhancedLinearPerceptron", 'tmp.csv' using 1:3 with lines linetype 2 title "LinearPerceptronEnsemble", 'tmp.csv' using 1:4 with lines linetype 3 title "MultilayerPerceptron"
EOH

  open($fh, '>', 'plot') or die "$!\n";
  print $fh $plot;
  close($fh);

  system("gnuplot plot");
}

sub runtime
{
  my $data = '';
  # produce the data to plot
  for my $i (1..9)
  {
    my $elp = 0;
    my $elps = 0;
    my $elpm = 0;
    my $lpe  = 0;
    my $ml = 0;
    my $n = 0;
    for my $f (keys(%{$results->{cross}}))
    {
      ++$n;
      my $test = $results->{cross}{$f}{test}{$i};
      $elp += $results->{standard}{$f}{nonstandardized}{$i}{time} / $test;
      $elps += $results->{standard}{$f}{standardized}{$i}{time} / $test;
      $elpm += $results->{cross}{$f}{xtrain}{$i}{time} / $test;
      $lpe += $results->{compare}{$f}{lpe}{$i}{time} / $test;
      $ml += $results->{compare}{$f}{ml}{$i}{time} / $test;
    }
    $data .= $i * 10 . ',' . ($elp / $n) * 100 . ','. ($elps / $n) * 100 . ','. ($elpm / $n) * 100 . ',' . ($lpe / $n) * 100 . ','. ($ml / $n) * 100 . "\n";
  }

  open(my $fh, '>', 'tmp.csv') or die "$!\n";
  print $fh $data;
  close($fh);

  my $plot = <<'EOH';
set title 'Runtime to train classifiers'
set ylabel 'Build time (ms)'
set xlabel 'Train set size (%)'
set grid
set term png
set output 'runtime.png'
set datafile separator ","
plot 'tmp.csv' using 1:2 with lines linetype 1 title "EnhancedLinearPerceptron (without standardization)", 'tmp.csv' using 1:3 with lines linetype 2 title "EnhancedLinearPerceptron (with standardization)", 'tmp.csv' using 1:4 with lines linetype 3 title "EnhancedLinearPerceptron (with model selection)", 'tmp.csv' using 1:5 with lines linetype 4 title "LinearPerceptronEnsemble", 'tmp.csv' using 1:6 with lines linetype 5 title "MultilayerPerceptron"
EOH

  open($fh, '>', 'plot') or die "$!\n";
  print $fh $plot;
  close($fh);

  system("gnuplot plot");
}

