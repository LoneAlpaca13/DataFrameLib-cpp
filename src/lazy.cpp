#include "lazy.h"

#include "utils.h"

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

EagerDataFrame LazyDataFrame::collect() const {
  auto table = readCSV(csv_path);
  EagerDataFrame df(table);

  // temp storage for group_by
  std::vector<std::string> group_keys;
  bool has_group = false;

  // Step 2: apply operations
  for (const auto& op : ops) {
    if (op.type == LazyOpType::FILTER) {
      df = df.filter(op.expr);
    }

    else if (op.type == LazyOpType::SELECT) {
      df = df.select(op.columns);
    }

    else if (op.type == LazyOpType::WITH_COLUMN) {
      df = df.with_column(op.column_name, op.expr);
    }

    else if (op.type == LazyOpType::SORT) {
      // only using first column for now
      df = df.sort(op.columns[0]);
    }

    else if (op.type == LazyOpType::GROUP_BY) {
      group_keys = op.columns;
      has_group = true;
    }

    else if (op.type == LazyOpType::AGGREGATE) {
      if (!has_group) {
        throw std::runtime_error("aggregate() called without group_by()");
      }

      auto grouped = df.group_by(group_keys[0]);

      // for now: only one aggregation
      auto pair = op.agg_map[0];

      df = grouped.aggregate(pair.first, pair.second);

      has_group = false;
    }
  }

  return df;
}