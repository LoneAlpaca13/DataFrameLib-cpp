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

  auto maybe_reader = arrow::csv::TableReader::Make(
      io_context, input, read_options, parse_options, convert_options);

  auto reader = maybe_reader.ValueOrDie();
  return reader->Read().ValueOrDie();
}

int main() {
  auto table = readCSV("../data.csv");
  EagerDataFrame df(table);

  std::cout << "=== Original ===\n";
  df.printHead(10);

  // 1️⃣ Arithmetic test
  auto add_expr = std::make_shared<AddExpr>(
      std::make_shared<ColumnExpr>("salary"),
      std::make_shared<LiteralExpr>(
          std::make_shared<arrow::Int64Scalar>(1000)));

  auto df_add = df.with_column("salary_plus_1000", add_expr);

  std::cout << "\n=== salary + 1000 ===\n";
  df_add.printHead(10);

  // 2️⃣ Comparison test
  auto cmp_expr = std::make_shared<BinaryExpr>(
      std::make_shared<ColumnExpr>("age"),
      std::make_shared<LiteralExpr>(std::make_shared<arrow::Int64Scalar>(30)),
      OpType::GT);

  std::cout << "\n=== age > 30 (raw result) ===\n";
  auto cmp_result = cmp_expr->evaluate(df.getTable());
  for (auto& v : cmp_result) {
    std::cout << v->ToString() << " ";
  }
  std::cout << "\n";

  // 3️⃣ Filter test
  auto df_filtered = df.filter(cmp_expr);

  std::cout << "\n=== Filter age > 30 ===\n";
  df_filtered.printHead(10);

  // 4️⃣ Combined expression test
  auto complex_expr = std::make_shared<BinaryExpr>(
      std::make_shared<BinaryExpr>(
          std::make_shared<ColumnExpr>("age"),
          std::make_shared<LiteralExpr>(
              std::make_shared<arrow::Int64Scalar>(30)),
          OpType::GT),
      std::make_shared<BinaryExpr>(
          std::make_shared<ColumnExpr>("salary"),
          std::make_shared<LiteralExpr>(
              std::make_shared<arrow::Int64Scalar>(55000)),
          OpType::GT),
      OpType::AND);

  auto df_complex = df.filter(complex_expr);

  std::cout << "\n=== age > 30 AND salary > 55000 ===\n";
  df_complex.printHead(10);

  return 0;
}