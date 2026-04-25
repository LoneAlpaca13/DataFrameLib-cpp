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

  auto expr = std::make_shared<GreaterExpr>(
      std::make_shared<ColumnExpr>("age"),
      std::make_shared<LiteralExpr>(std::make_shared<arrow::Int64Scalar>(30)));

  auto result = expr->evaluate(df.getTable());

  std::cout << "\n=== Expression Result (age > 30) ===\n";
  for (auto& val : result) {
    std::cout << val->ToString() << " ";
  }
  std::cout << "\n";

  return 0;
}