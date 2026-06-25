#pragma once
#include <memory>
#include <string>
#include <vector>

#include "dataframe.h"
#include "expr.h"

namespace dataframelib {

enum class SourceType { CSV, PARQUET };

enum class LazyOpType {
  FILTER,
  SELECT,
  WITH_COLUMN,
  GROUP_BY,
  AGGREGATE,
  SORT,
  HEAD,
  JOIN,
};

struct Operation {
  LazyOpType type;
  std::vector<std::string> columns;
  std::shared_ptr<Expr> expr;
  std::string column_name;
  std::vector<std::pair<std::string, std::string>> agg_map;
  int n = 0;
  bool ascending = true;
  std::shared_ptr<class LazyDataFrame> join_df;
  std::vector<std::string> join_keys;
  std::string join_how;
};

class LazyDataFrame {
 private:
  std::string source_path_;
  SourceType source_type_;
  std::vector<Operation> ops_;

 public:
  LazyDataFrame(const std::string& path, SourceType type);

  LazyDataFrame filter(ExprPtr expr) const;
  LazyDataFrame select(const std::vector<std::string>& cols) const;
  LazyDataFrame with_column(const std::string& name, ExprPtr expr) const;
  LazyDataFrame group_by(const std::vector<std::string>& keys) const;
  LazyDataFrame aggregate(
      const std::vector<std::pair<std::string, std::string>>& agg_map) const;
  // Only vector sort to avoid ambiguity with brace-init lists
  LazyDataFrame sort(const std::vector<std::string>& cols,
                     bool ascending = true) const;
  LazyDataFrame head(int n) const;
  LazyDataFrame join(const LazyDataFrame& other,
                     const std::vector<std::string>& keys,
                     const std::string& how) const;

  EagerDataFrame collect() const;
  void sink_csv(const std::string& path) const;
  void sink_parquet(const std::string& path) const;
  void explain(const std::string& path) const;

  const std::vector<Operation>& getOps() const { return ops_; }
  std::string getSourcePath() const { return source_path_; }
  SourceType getSourceType() const { return source_type_; }
};

}  // namespace dataframelib