#include <iostream>

#include "expr.h"
#include "lazy.h"

int main() {
  LazyDataFrame df("../data.csv");

  auto result = df.select({"name", "salary"})
                    .filter(std::make_shared<BinaryExpr>(
                        std::make_shared<ColumnExpr>("age"),
                        std::make_shared<LiteralExpr>(
                            std::make_shared<arrow::Int64Scalar>(30)),
                        OpType::GT))
                    .collect();

  result.printHead(10);

  return 0;
}