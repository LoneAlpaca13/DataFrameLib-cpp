#include <arrow/api.h>
#include <arrow/csv/api.h>
#include <arrow/csv/writer.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>

#include "dataframelib/dataframelib.h"

namespace dataframelib {

EagerDataFrame read_csv(const std::string& path) {
  auto input = arrow::io::ReadableFile::Open(path).ValueOrDie();

  auto read_options = arrow::csv::ReadOptions::Defaults();
  auto parse_options = arrow::csv::ParseOptions::Defaults();
  auto convert_options = arrow::csv::ConvertOptions::Defaults();

  arrow::io::IOContext io_context = arrow::io::default_io_context();

  auto reader = arrow::csv::TableReader::Make(io_context, input, read_options,
                                              parse_options, convert_options)
                    .ValueOrDie();

  auto table = reader->Read().ValueOrDie();

  return EagerDataFrame(table);
}

#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>

EagerDataFrame read_parquet(const std::string& path) {
  auto infile = arrow::io::ReadableFile::Open(path).ValueOrDie();

  auto reader = parquet::arrow::OpenFile(infile, arrow::default_memory_pool())
                    .ValueOrDie();

  auto table = reader->ReadTable().ValueOrDie();

  return EagerDataFrame(table);
}

LazyDataFrame scan_csv(const std::string& path) { return LazyDataFrame(path); }

LazyDataFrame scan_parquet(const std::string& path) {
  return LazyDataFrame(path);  // you can refine later
}

EagerDataFrame from_columns(
    const std::unordered_map<std::string, std::shared_ptr<arrow::Array>>&
        cols) {
  std::vector<std::shared_ptr<arrow::Field>> fields;
  std::vector<std::shared_ptr<arrow::Array>> arrays;

  for (const auto& [name, array] : cols) {
    fields.push_back(arrow::field(name, array->type()));
    arrays.push_back(array);
  }

  auto schema = std::make_shared<arrow::Schema>(fields);
  auto table = arrow::Table::Make(schema, arrays);

  return EagerDataFrame(table);
}

}  // namespace dataframelib