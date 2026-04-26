#include <iostream>

#include "expr.h"
#include "lazy.h"

int main() {
  LazyDataFrame df("../data.csv");

  std::cout << "\n=== Test 1: filter + select ===\n";

  auto result1 = df.filter(std::make_shared<BinaryExpr>(
                               std::make_shared<ColumnExpr>("age"),
                               std::make_shared<LiteralExpr>(
                                   std::make_shared<arrow::Int64Scalar>(30)),
                               OpType::GT))
                     .select({"name", "salary"})
                     .collect();

  result1.printHead(10);

  std::cout << "\n=== Test 2: with_column ===\n";

  auto result2 =
      df.with_column("bonus",
                     std::make_shared<BinaryExpr>(
                         std::make_shared<ColumnExpr>("salary"),
                         std::make_shared<LiteralExpr>(
                             std::make_shared<arrow::Int64Scalar>(1000)),
                         OpType::ADD))
          .collect();

  result2.printHead(10);

  // Test 3: sort
  std::cout << "\n=== Test 3: sort ===\n";

  auto result3 = df.sort({"age"}, true).collect();

  result3.printHead(10);

  // Test 4: group_by + aggregate
  std::cout << "\n=== Test 4: group_by + aggregate ===\n";

  auto result4 = df.group_by({"age"}).aggregate({{"salary", "sum"}}).collect();

  result4.printHead(10);

  return 0;
}