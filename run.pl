#!/usr/bin/perl

use warnings;
use strict;

opendir(my $dh, 'data') or die "$!\n";
my @files = grep(/\.arff$/, readdir($dh));
closedir($dh);

foreach (@files)
{
  my $output = `/usr/java/jdk1.8.0_131/bin/java -jar ML.jar data/$_`;
  open(my $fh, '>', "results/$_.csv") or die "$!\n";
  print $fh $output;
  close($fh);
}