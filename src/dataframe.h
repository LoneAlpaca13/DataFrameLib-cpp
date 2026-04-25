#pragma once
#include <arrow/api.h>

#include <memory>
#include <string>

class EagerDataFrame {
 private:
  std::shared_ptr<arrow::Table> table;

 public:
  EagerDataFrame(std::shared_ptr<arrow::Table> t);

  void printSchema() const;
  void printHead(int n = 5) const;

  std::shared_ptr<arrow::Table> getTable() const;
};