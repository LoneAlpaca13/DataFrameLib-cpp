#include <arrow/api.h>
#include <arrow/csv/api.h>
#include <arrow/io/api.h>

#include <iostream>

#include "dataframe.h"
#include "expr.h"
#include "lazy.h"

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
  LazyDataFrame df("../data.csv");

  auto lazy =
      df.filter(std::make_shared<BinaryExpr>(
                    std::make_shared<ColumnExpr>("age"),
                    std::make_shared<LiteralExpr>(
                        std::make_shared<arrow::Int64Scalar>(30)),
                    OpType::GT))
          .select({"name", "salary"})
          .with_column("bonus",
                       std::make_shared<BinaryExpr>(
                           std::make_shared<ColumnExpr>("salary"),
                           std::make_shared<LiteralExpr>(
                               std::make_shared<arrow::Int64Scalar>(1000)),
                           OpType::ADD))
          .group_by({"age"})
          .aggregate({{"salary", "sum"}})
          .sort({"age"}, true);

  std::cout << "Lazy pipeline built successfully\n";

  return 0;
}