#pragma once

#include <arrow/api.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "expr.h"

namespace dataframelib {

class EagerDataFrame;
class Expr;

class GroupedDataFrame {
 private:
  std::shared_ptr<arrow::Table> table;
  std::map<std::string, std::vector<int>> groups;

 public:
  GroupedDataFrame(std::shared_ptr<arrow::Table> t,
                   std::map<std::string, std::vector<int>> g)
      : table(std::move(t)), groups(std::move(g)) {}

  void printGroups() const;

  EagerDataFrame aggregate(
      const std::vector<std::pair<std::string, std::string>>& agg_map) const;
};

class EagerDataFrame {
 private:
  std::shared_ptr<arrow::Table> table;

 public:
  explicit EagerDataFrame(std::shared_ptr<arrow::Table> t);

  std::shared_ptr<arrow::Table> getTable() const;

  void printSchema() const;
  void printHead(int n = 5) const;

  EagerDataFrame select(const std::vector<std::shared_ptr<Expr>>& exprs) const;
  EagerDataFrame head(int n) const;

  EagerDataFrame filter(std::shared_ptr<Expr> predicate) const;
  EagerDataFrame with_column(const std::string& name,
                             std::shared_ptr<Expr> expr) const;

  EagerDataFrame sort(const std::string& column_name, bool asc = true) const;

  GroupedDataFrame group_by(const std::vector<std::string>& column_names) const;

  EagerDataFrame join(const EagerDataFrame& other,
                      const std::vector<std::string>& column_names,
                      const std::string& how) const;

  void write_csv(const std::string& path) const;
  void write_parquet(const std::string& path) const;
};
}  // namespace dataframelib