#include "dataframe.h"

#include <arrow/csv/writer.h>
#include <arrow/io/api.h>
#include <parquet/arrow/writer.h>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <unordered_map>

namespace dataframelib {

// ================= CONSTRUCTOR =================

EagerDataFrame::EagerDataFrame(std::shared_ptr<arrow::Table> t) : table(t) {}

std::shared_ptr<arrow::Table> EagerDataFrame::getTable() const { return table; }

// ================= PRINT =================

void EagerDataFrame::printSchema() const {
  std::cout << table->schema()->ToString() << std::endl;
}

void EagerDataFrame::printHead(int n) const {
  int rows = std::min(n, (int)table->num_rows());

  for (int j = 0; j < table->num_columns(); j++)
    std::cout << table->field(j)->name() << "\t";

  std::cout << "\n";

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < table->num_columns(); j++) {
      auto scalar = table->column(j)->chunk(0)->GetScalar(i).ValueOrDie();
      std::cout << scalar->ToString() << "\t";
    }
    std::cout << "\n";
  }
}

// ================= SELECT =================

EagerDataFrame EagerDataFrame::select(
    const std::vector<std::string>& columns) const {
  std::vector<std::shared_ptr<arrow::Field>> fields;
  std::vector<std::shared_ptr<arrow::ChunkedArray>> arrays;

  for (auto& name : columns) {
    int idx = table->schema()->GetFieldIndex(name);
    if (idx == -1) throw std::runtime_error("Column not found");

    fields.push_back(table->field(idx));
    arrays.push_back(table->column(idx));
  }

  return EagerDataFrame(
      arrow::Table::Make(std::make_shared<arrow::Schema>(fields), arrays));
}

// ================= HEAD =================

EagerDataFrame EagerDataFrame::head(int n) const {
  return EagerDataFrame(table->Slice(0, n));
}

// ================= FILTER =================

EagerDataFrame EagerDataFrame::filter(std::shared_ptr<Expr> predicate) const {
  auto mask = predicate->evaluate(table);

  std::vector<std::shared_ptr<arrow::Array>> new_arrays;

  for (int j = 0; j < table->num_columns(); j++) {
    auto chunk = table->column(j)->chunk(0);

    arrow::StringBuilder builder;

    for (int i = 0; i < chunk->length(); i++) {
      auto keep = std::dynamic_pointer_cast<arrow::BooleanScalar>(mask[i]);

      if (!keep || !keep->is_valid || !keep->value) continue;

      auto val = chunk->GetScalar(i).ValueOrDie();
      if (!val->is_valid)
        builder.AppendNull();
      else
        builder.Append(val->ToString());
    }

    std::shared_ptr<arrow::Array> arr;
    builder.Finish(&arr);
    new_arrays.push_back(arr);
  }

  return EagerDataFrame(arrow::Table::Make(table->schema(), new_arrays));
}

// ================= WITH COLUMN =================

EagerDataFrame EagerDataFrame::with_column(const std::string& name,
                                           std::shared_ptr<Expr> expr) const {
  auto values = expr->evaluate(table);

  if (values.empty()) throw std::runtime_error("Empty expression result");

  arrow::StringBuilder builder;

  for (auto& v : values) {
    if (!v->is_valid)
      builder.AppendNull();
    else
      builder.Append(v->ToString());
  }

  std::shared_ptr<arrow::Array> arr;
  builder.Finish(&arr);

  auto chunked = std::make_shared<arrow::ChunkedArray>(arr);

  int idx = table->schema()->GetFieldIndex(name);

  std::shared_ptr<arrow::Table> new_table;

  if (idx == -1) {
    new_table = table
                    ->AddColumn(table->num_columns(),
                                arrow::field(name, arrow::utf8()), chunked)
                    .ValueOrDie();
  } else {
    new_table =
        table->SetColumn(idx, arrow::field(name, arrow::utf8()), chunked)
            .ValueOrDie();
  }

  return EagerDataFrame(new_table);
}

// ================= SORT =================

EagerDataFrame EagerDataFrame::sort(const std::string& column_name) const {
  int idx = table->schema()->GetFieldIndex(column_name);
  if (idx == -1) throw std::runtime_error("Column not found");

  auto chunk = table->column(idx)->chunk(0);

  std::vector<int> indices(chunk->length());
  std::iota(indices.begin(), indices.end(), 0);

  std::sort(indices.begin(), indices.end(), [&](int a, int b) {
    auto A = chunk->GetScalar(a).ValueOrDie();
    auto B = chunk->GetScalar(b).ValueOrDie();

    if (!A->is_valid) return false;
    if (!B->is_valid) return true;

    return A->ToString() < B->ToString();
  });

  std::vector<std::shared_ptr<arrow::ChunkedArray>> new_cols;

  for (int j = 0; j < table->num_columns(); j++) {
    auto col = table->column(j)->chunk(0);

    arrow::StringBuilder builder;

    for (int i : indices) {
      auto scalar = col->GetScalar(i).ValueOrDie();

      if (!scalar->is_valid)
        builder.AppendNull();
      else
        builder.Append(scalar->ToString());
    }

    std::shared_ptr<arrow::Array> arr;
    builder.Finish(&arr);

    new_cols.push_back(std::make_shared<arrow::ChunkedArray>(arr));
  }

  return EagerDataFrame(arrow::Table::Make(table->schema(), new_cols));
}

// ================= GROUP BY =================

GroupedDataFrame EagerDataFrame::group_by(
    const std::vector<std::string>& column_names) const {
  int idx = table->schema()->GetFieldIndex(column_names[0]);
  auto chunk = table->column(idx)->chunk(0);

  std::map<std::string, std::vector<int>> groups;

  for (int i = 0; i < chunk->length(); i++) {
    auto scalar = chunk->GetScalar(i).ValueOrDie();
    if (!scalar->is_valid) continue;

    groups[scalar->ToString()].push_back(i);
  }

  return GroupedDataFrame(table, groups);
}

// ================= AGGREGATE =================

EagerDataFrame GroupedDataFrame::aggregate(const std::string& column_name,
                                           const std::string& op) const {
  int idx = table->schema()->GetFieldIndex(column_name);
  auto chunk = table->column(idx)->chunk(0);

  arrow::StringBuilder key_builder;
  arrow::Int64Builder val_builder;

  for (auto& [key, rows] : groups) {
    bool found = false;
    int64_t result = 0;

    if (op == "count") {
      int count = 0;
      for (int i : rows) {
        auto s = chunk->GetScalar(i).ValueOrDie();
        if (s->is_valid) count++;
      }
      result = count;
      found = true;
    }

    else if (op == "sum" || op == "mean" || op == "min" || op == "max") {
      int64_t sum = 0;
      int count = 0;
      int64_t minv = 0, maxv = 0;

      for (int i : rows) {
        auto s = chunk->GetScalar(i).ValueOrDie();
        if (!s->is_valid) continue;

        auto val = std::dynamic_pointer_cast<arrow::Int64Scalar>(s);
        if (!val) continue;

        int64_t v = val->value;

        if (!found) {
          minv = maxv = v;
          found = true;
        }

        sum += v;
        count++;

        if (v < minv) minv = v;
        if (v > maxv) maxv = v;
      }

      if (op == "sum")
        result = sum;
      else if (op == "mean" && count > 0)
        result = sum / count;
      else if (op == "min")
        result = minv;
      else if (op == "max")
        result = maxv;
    }

    key_builder.Append(key);

    if (!found)
      val_builder.AppendNull();
    else
      val_builder.Append(result);
  }

  std::shared_ptr<arrow::Array> keys, vals;
  key_builder.Finish(&keys);
  val_builder.Finish(&vals);

  return EagerDataFrame(arrow::Table::Make(
      arrow::schema({arrow::field("group_key", arrow::utf8()),
                     arrow::field(column_name + "_" + op, arrow::int64())}),
      {std::make_shared<arrow::ChunkedArray>(keys),
       std::make_shared<arrow::ChunkedArray>(vals)}));
}

// ================= WRITE =================

void EagerDataFrame::write_csv(const std::string& path) const {
  auto output = arrow::io::FileOutputStream::Open(path).ValueOrDie();
  auto options = arrow::csv::WriteOptions::Defaults();

  auto status = arrow::csv::WriteCSV(*table, options, output.get());

  if (!status.ok()) {
    throw std::runtime_error(status.ToString());
  }
}

void EagerDataFrame::write_parquet(const std::string& path) const {
  auto outfile = arrow::io::FileOutputStream::Open(path).ValueOrDie();

  // create writer (modern API)
  auto writer_result = parquet::arrow::FileWriter::Open(
      *table->schema(), arrow::default_memory_pool(), outfile);

  if (!writer_result.ok()) {
    throw std::runtime_error(writer_result.status().ToString());
  }

  std::unique_ptr<parquet::arrow::FileWriter> writer =
      std::move(writer_result.ValueOrDie());

  // write table
  auto status = writer->WriteTable(*table, table->num_rows());
  if (!status.ok()) {
    throw std::runtime_error(status.ToString());
  }

  // close
  status = writer->Close();
  if (!status.ok()) {
    throw std::runtime_error(status.ToString());
  }
}

}  // namespace dataframelib