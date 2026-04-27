#pragma once
#include <memory>
#include <string>
#include <vector>

#include "dataframe.h"
#include "expr.h"

namespace dataframelib {

// ================= OP TYPES =================

enum class LazyOpType {
  FILTER,
  SELECT,
  WITH_COLUMN,
  GROUP_BY,
  AGGREGATE,
  SORT,
  HEAD
};

// ================= OP STRUCT =================

struct Operation {
  LazyOpType type;

  std::vector<std::string> columns;
  std::shared_ptr<Expr> expr;
  std::string column_name;

  std::vector<std::pair<std::string, std::string>> agg_map;

  int n = 0;  // used for HEAD
};

// ================= CLASS =================

class LazyDataFrame {
 private:
  std::string csv_path;
  std::vector<Operation> ops;

 public:
  LazyDataFrame(const std::string& path);

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

  // execution
  EagerDataFrame collect() const;

  // debugging
  void explain(const std::string& path) const;
};

}  // namespace dataframelib