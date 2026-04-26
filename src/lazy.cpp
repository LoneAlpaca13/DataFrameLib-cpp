#include "lazy.h"

LazyDataFrame::LazyDataFrame(const std::string& path) : csv_path(path) {}

LazyDataFrame LazyDataFrame::filter(std::shared_ptr<Expr> expr) const {
  LazyDataFrame df = *this;

  Operation op;
  op.type = LazyOpType::FILTER;
  op.expr = expr;

  df.ops.push_back(op);
  return df;
}

LazyDataFrame LazyDataFrame::select(
    const std::vector<std::string>& cols) const {
  LazyDataFrame df = *this;

  Operation op;
  op.type = LazyOpType::SELECT;
  op.columns = cols;

  df.ops.push_back(op);
  return df;
}

LazyDataFrame LazyDataFrame::with_column(const std::string& name,
                                         std::shared_ptr<Expr> expr) const {
  LazyDataFrame df = *this;

  Operation op;
  op.type = LazyOpType::WITH_COLUMN;
  op.column_name = name;
  op.expr = expr;

  df.ops.push_back(op);
  return df;
}

LazyDataFrame LazyDataFrame::group_by(
    const std::vector<std::string>& keys) const {
  LazyDataFrame df = *this;

  Operation op;
  op.type = LazyOpType::GROUP_BY;
  op.columns = keys;

  df.ops.push_back(op);
  return df;
}

LazyDataFrame LazyDataFrame::aggregate(
    const std::vector<std::pair<std::string, std::string>>& agg_map) const {
  LazyDataFrame df = *this;

  Operation op;
  op.type = LazyOpType::AGGREGATE;
  op.agg_map = agg_map;

  df.ops.push_back(op);
  return df;
}

LazyDataFrame LazyDataFrame::sort(const std::vector<std::string>& cols,
                                  bool asc) const {
  LazyDataFrame df = *this;

  Operation op;
  op.type = LazyOpType::SORT;
  op.columns = cols;
  // ignoring asc for now (store later if needed)

  df.ops.push_back(op);
  return df;
}