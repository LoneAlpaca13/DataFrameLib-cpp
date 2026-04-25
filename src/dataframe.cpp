#include "dataframe.h"

#include <iostream>

EagerDataFrame::EagerDataFrame(std::shared_ptr<arrow::Table> t) : table(t) {}

std::shared_ptr<arrow::Table> EagerDataFrame::getTable() const { return table; }

void EagerDataFrame::printSchema() const {
  std::cout << table->schema()->ToString() << std::endl;
}

void EagerDataFrame::printHead(int n) const {
  int rows = std::min(n, (int)table->num_rows());
  int cols = table->num_columns();

  // header
  for (int j = 0; j < cols; j++) {
    std::cout << table->field(j)->name() << "\t";
  }
  std::cout << "\n";

  // rows
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      auto col = table->column(j);
      auto chunk = col->chunk(0);  // assume single chunk for now

      auto scalar = chunk->GetScalar(i).ValueOrDie();
      std::cout << scalar->ToString() << "\t";
    }
    std::cout << "\n";
  }
}

EagerDataFrame EagerDataFrame::select(
    const std::vector<std::string>& columns) const {
  std::vector<std::shared_ptr<arrow::Field>> fields;
  std::vector<std::shared_ptr<arrow::ChunkedArray>> arrays;

  for (const auto& name : columns) {
    int idx = table->schema()->GetFieldIndex(name);

    if (idx == -1) {
      throw std::runtime_error("Column not found: " + name);
    }

    fields.push_back(table->field(idx));
    arrays.push_back(table->column(idx));  // zero-copy
  }

  auto schema = std::make_shared<arrow::Schema>(fields);
  auto new_table = arrow::Table::Make(schema, arrays);

  return EagerDataFrame(new_table);
}

EagerDataFrame EagerDataFrame::head(int n) const {
  int rows = std::min(n, (int)table->num_rows());
  auto new_table = table->Slice(0, rows);  // zero-copy slice
  return EagerDataFrame(new_table);
}