#pragma once
#include <arrow/api.h>
#include <map>
#include <unordered_map>
#include <memory>
#include <string>
#include <vector>
#include "expr.h"

namespace dataframelib {

class EagerDataFrame;

class GroupedDataFrame {
 private:
  std::shared_ptr<arrow::Table> table_;
  std::vector<std::string> keys_;
  std::map<std::string, std::vector<int>> groups_;
 public:
  GroupedDataFrame(std::shared_ptr<arrow::Table> t,
                   std::vector<std::string> keys,
                   std::map<std::string, std::vector<int>> g)
      : table_(std::move(t)), keys_(std::move(keys)), groups_(std::move(g)) {}
  EagerDataFrame aggregate(
      const std::vector<std::pair<std::string, std::string>>& agg_map) const;
};

class EagerDataFrame {
 private:
  std::shared_ptr<arrow::Table> table_;
 public:
  explicit EagerDataFrame(std::shared_ptr<arrow::Table> t);
  std::shared_ptr<arrow::Table> getTable() const;
  int64_t num_rows() const;
  int num_columns() const;
  void printSchema() const;
  void printHead(int n = 5) const;
  EagerDataFrame select(const std::vector<std::string>& col_names) const;
  EagerDataFrame filter(ExprPtr predicate) const;
  EagerDataFrame with_column(const std::string& name, ExprPtr expr) const;
  EagerDataFrame sort(const std::vector<std::string>& column_names,
                      bool ascending = true) const;
  EagerDataFrame sort_head(const std::vector<std::string>& column_names,
                           bool ascending, int64_t n) const;
  EagerDataFrame head(int n) const;
  GroupedDataFrame group_by(const std::vector<std::string>& column_names) const;
  EagerDataFrame join(const EagerDataFrame& other,
                      const std::vector<std::string>& column_names,
                      const std::string& how) const;
  void write_csv(const std::string& path) const;
  void write_parquet(const std::string& path) const;
};

}  // namespace dataframelib
