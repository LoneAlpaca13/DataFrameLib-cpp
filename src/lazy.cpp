#include "lazy.h"

#include "utils.h"

std::string getFilterColumn(const std::shared_ptr<Expr>& expr) {
  auto col = std::dynamic_pointer_cast<ColumnExpr>(expr);
  if (col) return col->getName();

  auto bin = std::dynamic_pointer_cast<BinaryExpr>(expr);
  if (bin) {
    auto left = std::dynamic_pointer_cast<ColumnExpr>(bin->getLeft());
    if (left) return left->getName();
  }

  return "";
}

std::vector<Operation> optimize(const std::vector<Operation>& ops) {
  std::vector<Operation> result;

  for (size_t i = 0; i < ops.size(); i++) {
    const auto& op = ops[i];

    if (op.type == LazyOpType::FILTER) {
      std::string filter_col = getFilterColumn(op.expr);

      size_t j = result.size();

      while (j > 0) {
        const auto& prev = result[j - 1];

        bool can_swap = true;

        if (prev.type == LazyOpType::WITH_COLUMN) {
          if (prev.column_name == filter_col) {
            can_swap = false;
          }
        }

        if (prev.type == LazyOpType::GROUP_BY ||
            prev.type == LazyOpType::AGGREGATE) {
          can_swap = false;
        }

        if (!can_swap) break;

        j--;
      }

      result.insert(result.begin() + j, op);
    }

    else {
      result.push_back(op);
    }
  }

  return result;
}

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

  std::vector<std::string> group_keys;
  bool has_group = false;

  auto optimized_ops = optimize(ops);

  for (const auto& op : optimized_ops) {
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