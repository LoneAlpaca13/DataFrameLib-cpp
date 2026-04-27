#include <arrow/api.h>
#include <arrow/csv/api.h>
#include <arrow/csv/writer.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>

#include "dataframelib/dataframelib.h"

namespace dataframelib {

// ================= READ =================

EagerDataFrame read_csv(const std::string& path) {
  auto input = arrow::io::ReadableFile::Open(path).ValueOrDie();

  auto reader =
      arrow::csv::TableReader::Make(arrow::io::default_io_context(), input,
                                    arrow::csv::ReadOptions::Defaults(),
                                    arrow::csv::ParseOptions::Defaults(),
                                    arrow::csv::ConvertOptions::Defaults())
          .ValueOrDie();

  auto table = reader->Read().ValueOrDie();

  return EagerDataFrame(table);
}

EagerDataFrame read_parquet(const std::string& path) {
  auto infile = arrow::io::ReadableFile::Open(path).ValueOrDie();

  auto reader = parquet::arrow::OpenFile(infile, arrow::default_memory_pool())
                    .ValueOrDie();

  auto table = reader->ReadTable().ValueOrDie();

  return EagerDataFrame(table);
}

// ================= SCAN =================

LazyDataFrame scan_csv(const std::string& path) { return LazyDataFrame(path); }

LazyDataFrame scan_parquet(const std::string& path) {
  return LazyDataFrame(path);
}

// ================= FROM COLUMNS =================

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

// ================= SINK =================

void sink_csv(const LazyDataFrame& df, const std::string& path) {
  auto result = df.collect();
  auto table = result.getTable();

  auto output = arrow::io::FileOutputStream::Open(path).ValueOrDie();

  auto options = arrow::csv::WriteOptions::Defaults();

  auto status = arrow::csv::WriteCSV(*table, options, output.get());

  if (!status.ok()) {
    throw std::runtime_error(status.ToString());
  }
}

void sink_parquet(const LazyDataFrame& df, const std::string& path) {
  auto result = df.collect();
  auto table = result.getTable();

  auto outfile = arrow::io::FileOutputStream::Open(path).ValueOrDie();

  // modern Arrow API
  auto writer_result = parquet::arrow::FileWriter::Open(
      *table->schema(), arrow::default_memory_pool(), outfile);

  if (!writer_result.ok()) {
    throw std::runtime_error(writer_result.status().ToString());
  }

  std::unique_ptr<parquet::arrow::FileWriter> writer =
      std::move(writer_result.ValueOrDie());

  auto status = writer->WriteTable(*table, table->num_rows());
  if (!status.ok()) {
    throw std::runtime_error(status.ToString());
  }

  status = writer->Close();
  if (!status.ok()) {
    throw std::runtime_error(status.ToString());
  }
}

}  // namespace dataframelib