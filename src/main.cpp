#include <iostream>

#include "dataframelib/dataframelib.h"
using namespace dataframelib;
int main() {
  auto df = read_csv("../data.csv");

  // length test
  auto df2 =
      df.with_column("len", length(std::make_shared<ColumnExpr>("name")));

  // contains test
  auto df3 =
      df.filter(contains(std::make_shared<ColumnExpr>("name"),
                         std::make_shared<LiteralExpr>(
                             std::make_shared<arrow::StringScalar>("a"))));

  df2.printHead();
  df3.printHead();
  return 0;
}