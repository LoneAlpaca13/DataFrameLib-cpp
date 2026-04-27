#include <iostream>

#include "dataframelib/dataframelib.h"
using namespace dataframelib;
int main() {
  auto df = read_csv("data.csv");

  auto new_df =
      df.with_column("abs_col", abs(std::make_shared<ColumnExpr>("col")));

  new_df.printHead();
  return 0;
}