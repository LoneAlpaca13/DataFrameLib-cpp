#pragma once
#include <memory>
#include <string>
#include <vector>

#include "dataframe.h"
#include "expr.h"

namespace dataframelib {

// ================= SOURCE TYPE =================
enum class SourceType { CSV, PARQUET };

// ================= OP TYPES =================
enum class LazyOpType {
  FILTER,
  SELECT,
  WITH_COLUMN,
  GROUP_BY,
  AGGREGATE,
  SORT,
  HEAD,
  JOIN  // ✅ added
};

// ================= OP STRUCT =================
struct Operation {
  LazyOpType type;

  std::vector<std::string> columns;
  std::shared_ptr<Expr> expr;
  std::string column_name;

  std::vector<std::pair<std::string, std::string>> agg_map;

  int n = 0;

  // ✅ JOIN SUPPORT (CORRECT PLACE)
  class LazyDataFrame* join_df = nullptr;
  std::vector<std::string> join_keys;
  std::string join_how;
};

// ================= CLASS =================
class LazyDataFrame {
 private:
  std::string csv_path;
  std::vector<Operation> ops;

  SourceType source_type;

 public:
  // ✅ correct constructor
  LazyDataFrame(const std::string& path, SourceType type);

  // operations
  LazyDataFrame filter(std::shared_ptr<Expr> expr) const;
  LazyDataFrame select(const std::vector<std::string>& cols) const;
  LazyDataFrame with_column(const std::string& name,
                            std::shared_ptr<Expr> expr) const;
  LazyDataFrame group_by(const std::vector<std::string>& keys) const;
  LazyDataFrame aggregate(
      const std::vector<std::pair<std::string, std::string>>& agg_map) const;
  LazyDataFrame sort(const std::vector<std::string>& cols, bool asc) const;
  LazyDataFrame head(int n) const;

  // ✅ ADD THIS (missing before)
  LazyDataFrame join(const LazyDataFrame& other,
                     const std::vector<std::string>& keys,
                     const std::string& how) const;

  // execution
  EagerDataFrame collect() const;

  // debugging
  void explain(const std::string& path) const;
};

}  // namespace dataframelib