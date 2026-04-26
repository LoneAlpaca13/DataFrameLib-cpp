#pragma once
#include <arrow/api.h>

#include <memory>
#include <string>
#include <vector>

#include "expr.h"

class EagerDataFrame {
 private:
  std::shared_ptr<arrow::Table> table;

 public:
  EagerDataFrame(std::shared_ptr<arrow::Table> t);

  std::shared_ptr<arrow::Table> getTable() const;

  void printSchema() const;
  void printHead(int n = 5) const;

  EagerDataFrame select(const std::vector<std::string>& columns) const;
  EagerDataFrame head(int n) const;

  EagerDataFrame filter(std::shared_ptr<Expr> predicate) const;

  EagerDataFrame with_column(const std::string& name,
                             std::shared_ptr<Expr> expr) const;
};