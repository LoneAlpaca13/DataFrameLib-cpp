#include "dataframe.h"

#include <iostream>

EagerDataFrame::EagerDataFrame(std::shared_ptr<arrow::Table> t) : table(t) {}

void EagerDataFrame::printSchema() const {
  std::cout << table->schema()->ToString() << std::endl;
}

void EagerDataFrame::printHead(int n) const {
  int rows = std::min(n, (int)table->num_rows());
  int cols = table->num_columns();

  // print header
  for (int j = 0; j < cols; j++) {
    std::cout << table->field(j)->name() << "\t";
  }
  std::cout << "\n";

  // print rows
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      auto column = table->column(j);
      auto chunk = column->chunk(0);

      auto scalar = chunk->GetScalar(i).ValueOrDie();
      std::cout << scalar->ToString() << "\t";
    }
    std::cout << "\n";
  }
}
std::shared_ptr<arrow::Table> EagerDataFrame::getTable() const { return table; }