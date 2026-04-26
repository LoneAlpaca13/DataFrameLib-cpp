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

  auto new_df = df.with_column(
      "salary_plus_1000", std::make_shared<AddExpr>(
                              std::make_shared<ColumnExpr>("salary"),
                              std::make_shared<LiteralExpr>(
                                  std::make_shared<arrow::Int64Scalar>(1000))));

  std::cout << "\n=== with_column test ===\n";
  new_df.printHead(10);
  return 0;
}