#include <iostream>

#include "dataframelib/dataframelib.h"
using namespace dataframelib;
int main() {
  auto df = dataframelib::read_csv("../data.csv");

  df.printSchema();
  df.printHead();

  return 0;
}