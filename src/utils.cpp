#include <arrow/api.h>
#include <arrow/csv/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>
#include <stdexcept>
#include <vector>
#include "dataframelib/dataframelib.h"
namespace dataframelib {
EagerDataFrame read_csv(const std::string& path) {
  auto input = arrow::io::ReadableFile::Open(path).ValueOrDie();
  auto reader = arrow::csv::TableReader::Make(arrow::io::default_io_context(), input,
    arrow::csv::ReadOptions::Defaults(), arrow::csv::ParseOptions::Defaults(),
    arrow::csv::ConvertOptions::Defaults()).ValueOrDie();
  return EagerDataFrame(reader->Read().ValueOrDie());
}
EagerDataFrame read_parquet(const std::string& path) {
  auto infile = arrow::io::ReadableFile::Open(path).ValueOrDie();
  auto reader = parquet::arrow::OpenFile(infile, arrow::default_memory_pool()).ValueOrDie();
  return EagerDataFrame(reader->ReadTable().ValueOrDie());
}
LazyDataFrame scan_csv(const std::string& path) { return LazyDataFrame(path, SourceType::CSV); }
LazyDataFrame scan_parquet(const std::string& path) { return LazyDataFrame(path, SourceType::PARQUET); }
EagerDataFrame from_columns(const std::vector<std::pair<std::string, std::shared_ptr<arrow::Array>>>& cols) {
  std::vector<std::shared_ptr<arrow::Field>> fields;
  std::vector<std::shared_ptr<arrow::ChunkedArray>> arrays;
  for (const auto& [name, arr] : cols) {
    fields.push_back(arrow::field(name, arr->type()));
    arrays.push_back(std::make_shared<arrow::ChunkedArray>(arr));
  }
  return EagerDataFrame(arrow::Table::Make(std::make_shared<arrow::Schema>(fields), arrays));
}
}
