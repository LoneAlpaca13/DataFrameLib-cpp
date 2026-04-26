#include "utils.h"

#include <arrow/csv/api.h>
#include <arrow/io/api.h>

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