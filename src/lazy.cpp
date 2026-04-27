#include "lazy.h"

#include <fstream>

#include "dataframelib/dataframelib.h"

namespace dataframelib {

// ================= HELPER =================
LazyDataFrame::LazyDataFrame(const std::string& path, SourceType type)
    : csv_path(path), source_type(type) {}

std::string getFilterColumn(const std::shared_ptr<Expr>& expr) {
  if (auto col = std::dynamic_pointer_cast<ColumnExpr>(expr))
    return col->getName();

  if (auto bin = std::dynamic_pointer_cast<BinaryExpr>(expr))
    return getFilterColumn(bin->getLeft());

  return "";
}

// ================= OPTIMIZER =================

std::vector<Operation> optimize(const std::vector<Operation>& ops) {
  std::vector<Operation> result;

  for (auto op : ops) {
    // ===== FILTER PUSHDOWN =====
    if (op.type == LazyOpType::FILTER) {
      size_t pos = result.size();

      while (pos > 0) {
        const auto& prev = result[pos - 1];

        if (prev.type == LazyOpType::GROUP_BY ||
            prev.type == LazyOpType::AGGREGATE)
          break;

        pos--;
      }

      result.insert(result.begin() + pos, op);
      continue;
    }

    // ===== SELECT PUSHDOWN =====
    if (op.type == LazyOpType::SELECT) {
      size_t pos = result.size();

      while (pos > 0) {
        const auto& prev = result[pos - 1];

        if (prev.type == LazyOpType::GROUP_BY ||
            prev.type == LazyOpType::AGGREGATE)
          break;

        pos--;
      }

      result.insert(result.begin() + pos, op);
      continue;
    }

    // ===== CONSTANT FOLDING (basic) =====
    if (op.type == LazyOpType::WITH_COLUMN) {
      auto bin = std::dynamic_pointer_cast<BinaryExpr>(op.expr);

      if (bin) {
        auto l = std::dynamic_pointer_cast<LiteralExpr>(bin->getLeft());
        auto r = std::dynamic_pointer_cast<LiteralExpr>(bin->getRight());

        if (l != nullptr && r != nullptr) {
          auto lv =
              std::dynamic_pointer_cast<arrow::Int64Scalar>(l->getValue());
          auto rv =
              std::dynamic_pointer_cast<arrow::Int64Scalar>(r->getValue());

          if (lv && rv) {
            int64_t val = lv->value + rv->value;
            op.expr = lit(val);
          }
        }
      }
    }

    // ===== HEAD PUSHDOWN =====
    if (op.type == LazyOpType::HEAD) {
      size_t pos = result.size();

      while (pos > 0) {
        const auto& prev = result[pos - 1];

        if (prev.type == LazyOpType::GROUP_BY ||
            prev.type == LazyOpType::AGGREGATE)
          break;

        pos--;
      }

      result.insert(result.begin() + pos, op);
      continue;
    }

    result.push_back(op);
  }

  return result;
}

// ================= COLLECT =================

EagerDataFrame LazyDataFrame::collect() const {
  EagerDataFrame df = (source_type == SourceType::CSV) ? read_csv(csv_path)
                                                       : read_parquet(csv_path);

  auto optimized_ops = optimize(ops);

  std::vector<std::string> group_keys;
  bool has_group = false;

  for (const auto& op : optimized_ops) {
    if (op.type == LazyOpType::FILTER) {
      df = df.filter(op.expr);
    }

    else if (op.type == LazyOpType::SELECT) {
      std::vector<std::shared_ptr<Expr>> exprs;
      for (const auto& c : op.columns) exprs.push_back(col(c));

      df = df.select(exprs);
    }

    else if (op.type == LazyOpType::WITH_COLUMN) {
      df = df.with_column(op.column_name, op.expr);
    }

    else if (op.type == LazyOpType::SORT) {
      df = df.sort(op.columns[0]);
    }

    else if (op.type == LazyOpType::HEAD) {
      df = df.head(op.n);
    }

    else if (op.type == LazyOpType::GROUP_BY) {
      group_keys = op.columns;
      has_group = true;
    }

    else if (op.type == LazyOpType::AGGREGATE) {
      if (!has_group)
        throw std::runtime_error("aggregate() without group_by()");

      auto grouped = df.group_by(group_keys);
      df = grouped.aggregate(op.agg_map);
      has_group = false;
    }

    else if (op.type == LazyOpType::JOIN) {
      auto right = op.join_df->collect();
      df = df.join(right, op.join_keys, op.join_how);
    }
  }

  return df;
}

// ================= EXPLAIN =================

void LazyDataFrame::explain(const std::string& path) const {
  std::ofstream out(path);

  out << "digraph G {\n";
  out << "rankdir=LR;\n";

  out << "node0 [label=\"SCAN\"];\n";

  auto optimized_ops = optimize(ops);

  for (size_t i = 0; i < optimized_ops.size(); i++) {
    const auto& op = optimized_ops[i];

    std::string label;

    switch (op.type) {
      case LazyOpType::FILTER:
        label = "FILTER";
        break;
      case LazyOpType::SELECT:
        label = "SELECT";
        break;
      case LazyOpType::WITH_COLUMN:
        label = "WITH_COLUMN";
        break;
      case LazyOpType::GROUP_BY:
        label = "GROUP_BY";
        break;
      case LazyOpType::AGGREGATE:
        label = "AGG";
        break;
      case LazyOpType::SORT:
        label = "SORT";
        break;
      case LazyOpType::HEAD:
        label = "HEAD";
        break;
      case LazyOpType::JOIN:
        label = "JOIN";
        break;
    }

    out << "node" << i + 1 << " [label=\"" << label << "\"];\n";
    out << "node" << i << " -> node" << i + 1 << ";\n";
  }

  out << "}\n";
}

}  // namespace dataframelib