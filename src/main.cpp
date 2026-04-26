#include <arrow/api.h>
#include <arrow/csv/api.h>
#include <arrow/io/api.h>

#include <iostream>

#include "dataframe.h"
#include "expr.h"

std::shared_ptr<arrow::Table> readCSV(const std::string& filename) {
  auto input = arrow::io::ReadableFile::Open(filename).ValueOrDie();

  auto read_options = arrow::csv::ReadOptions::Defaults();
  auto parse_options = arrow::csv::ParseOptions::Defaults();
  auto convert_options = arrow::csv::ConvertOptions::Defaults();

  arrow::io::IOContext io_context = arrow::io::default_io_context();

  auto reader = arrow::csv::TableReader::Make(io_context, input, read_options,
                                              parse_options, convert_options)
                    .ValueOrDie();

  return reader->Read().ValueOrDie();
}

int main() {
  auto table = readCSV("../data.csv");
  EagerDataFrame df(table);

  std::cout << "\n=== Original Data ===\n";
  df.printHead(10);

  auto add_expr = std::make_shared<BinaryExpr>(
      std::make_shared<ColumnExpr>("salary"),
      std::make_shared<LiteralExpr>(std::make_shared<arrow::Int64Scalar>(1000)),
      OpType::ADD);

  auto df_add = df.with_column("salary_plus_1000", add_expr);

  std::cout << "\n=== With Column (salary + 1000) ===\n";
  df_add.printHead(10);

  auto age_gt_30 = std::make_shared<BinaryExpr>(
      std::make_shared<ColumnExpr>("age"),
      std::make_shared<LiteralExpr>(std::make_shared<arrow::Int64Scalar>(30)),
      OpType::GT);

  auto df_filtered = df.filter(age_gt_30);

  std::cout << "\n=== Filter: age > 30 ===\n";
  df_filtered.printHead(10);

  // -------- 4. Complex Filter --------
  auto salary_gt_55000 = std::make_shared<BinaryExpr>(
      std::make_shared<ColumnExpr>("salary"),
      std::make_shared<LiteralExpr>(
          std::make_shared<arrow::Int64Scalar>(55000)),
      OpType::GT);

  auto complex_filter =
      std::make_shared<BinaryExpr>(age_gt_30, salary_gt_55000, OpType::AND);

  auto df_complex = df.filter(complex_filter);

  std::cout << "\n=== Filter: age > 30 AND salary > 55000 ===\n";
  df_complex.printHead(10);

  // -------- 5. Select --------
  auto df_select = df.select({"name", "salary"});

  std::cout << "\n=== Select (name, salary) ===\n";
  df_select.printHead(10);

  // -------- 6. Head --------
  auto df_head = df.head(3);

  std::cout << "\n=== Head (3 rows) ===\n";
  df_head.printHead(3);

  auto df_sorted = df.sort("age");

  std::cout << "\n=== Sorted by age ===\n";
  df_sorted.printHead(10);

  auto grouped = df.group_by("age");

  std::cout << "\n=== Groups by age ===\n";
  grouped.printGroups();

  return 0;
}