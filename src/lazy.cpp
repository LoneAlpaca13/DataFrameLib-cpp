#include "dataframelib/lazy.h"

#include <arrow/csv/writer.h>
#include <arrow/io/api.h>
#include <parquet/arrow/writer.h>

#include <fstream>
#include <stdexcept>

#include "dataframelib/dataframelib.h"

namespace dataframelib {

LazyDataFrame::LazyDataFrame(const std::string& path, SourceType type)
    : source_path_(path), source_type_(type) {}

LazyDataFrame LazyDataFrame::filter(ExprPtr expr) const {
  LazyDataFrame copy = *this;
  Operation op;
  op.type = LazyOpType::FILTER;
  op.expr = expr.ptr;
  copy.ops_.push_back(op);
  return copy;
}
LazyDataFrame LazyDataFrame::select(
    const std::vector<std::string>& cols) const {
  LazyDataFrame copy = *this;
  Operation op;
  op.type = LazyOpType::SELECT;
  op.columns = cols;
  copy.ops_.push_back(op);
  return copy;
}
LazyDataFrame LazyDataFrame::with_column(const std::string& name,
                                         ExprPtr expr) const {
  LazyDataFrame copy = *this;
  Operation op;
  op.type = LazyOpType::WITH_COLUMN;
  op.column_name = name;
  op.expr = expr.ptr;
  copy.ops_.push_back(op);
  return copy;
}
LazyDataFrame LazyDataFrame::group_by(
    const std::vector<std::string>& keys) const {
  LazyDataFrame copy = *this;
  Operation op;
  op.type = LazyOpType::GROUP_BY;
  op.columns = keys;
  copy.ops_.push_back(op);
  return copy;
}
LazyDataFrame LazyDataFrame::aggregate(
    const std::vector<std::pair<std::string, std::string>>& agg_map) const {
  LazyDataFrame copy = *this;
  Operation op;
  op.type = LazyOpType::AGGREGATE;
  op.agg_map = agg_map;
  copy.ops_.push_back(op);
  return copy;
}
LazyDataFrame LazyDataFrame::sort(const std::vector<std::string>& cols,
                                  bool ascending) const {
  LazyDataFrame copy = *this;
  Operation op;
  op.type = LazyOpType::SORT;
  op.columns = cols;
  op.ascending = ascending;
  copy.ops_.push_back(op);
  return copy;
}
LazyDataFrame LazyDataFrame::head(int n) const {
  LazyDataFrame copy = *this;
  Operation op;
  op.type = LazyOpType::HEAD;
  op.n = n;
  copy.ops_.push_back(op);
  return copy;
}
LazyDataFrame LazyDataFrame::join(const LazyDataFrame& other,
                                  const std::vector<std::string>& keys,
                                  const std::string& how) const {
  LazyDataFrame copy = *this;
  Operation op;
  op.type = LazyOpType::JOIN;
  op.join_df = std::make_shared<LazyDataFrame>(other);
  op.join_keys = keys;
  op.join_how = how;
  copy.ops_.push_back(op);
  return copy;
}

// ========================= EXPRESSION SIMPLIFICATION =========================
static std::shared_ptr<Expr> simplify_expr(std::shared_ptr<Expr> expr) {
  if (!expr) return expr;
  auto bin = std::dynamic_pointer_cast<BinaryExpr>(expr);
  if (bin) {
    auto left = simplify_expr(bin->getLeft());
    auto right = simplify_expr(bin->getRight());
    auto op = bin->getOp();
    auto rlit = std::dynamic_pointer_cast<LiteralExpr>(right);
    auto llit = std::dynamic_pointer_cast<LiteralExpr>(left);
    if (rlit && rlit->value && rlit->value->is_valid) {
      double rv = 0.0;
      if (rlit->value->type->id() == arrow::Type::DOUBLE)
        rv = std::static_pointer_cast<arrow::DoubleScalar>(rlit->value)->value;
      else if (rlit->value->type->id() == arrow::Type::INT64)
        rv = (double)std::static_pointer_cast<arrow::Int64Scalar>(rlit->value)
                 ->value;
      if (op == OpType::ADD && rv == 0.0) return left;
      if (op == OpType::SUB && rv == 0.0) return left;
      if (op == OpType::MUL && rv == 1.0) return left;
      if (op == OpType::MUL && rv == 0.0)
        return std::make_shared<LiteralExpr>(arrow::MakeScalar((int64_t)0));
      if (op == OpType::DIV && rv == 1.0) return left;
    }
    if (llit && rlit && llit->value && rlit->value && llit->value->is_valid &&
        rlit->value->is_valid) {
      auto gd = [](const std::shared_ptr<arrow::Scalar>& s) -> double {
        if (s->type->id() == arrow::Type::DOUBLE)
          return std::static_pointer_cast<arrow::DoubleScalar>(s)->value;
        if (s->type->id() == arrow::Type::INT64)
          return (double)std::static_pointer_cast<arrow::Int64Scalar>(s)->value;
        return 0.0;
      };
      double lv = gd(llit->value), rv = gd(rlit->value), res = 0.0;
      bool folded = true;
      switch (op) {
        case OpType::ADD:
          res = lv + rv;
          break;
        case OpType::SUB:
          res = lv - rv;
          break;
        case OpType::MUL:
          res = lv * rv;
          break;
        case OpType::DIV:
          res = (rv != 0.0) ? lv / rv : 0.0;
          break;
        default:
          folded = false;
      }
      if (folded) {
        bool both_int = llit->value->type->id() == arrow::Type::INT64 &&
                        rlit->value->type->id() == arrow::Type::INT64 &&
                        op != OpType::DIV;
        return both_int ? std::make_shared<LiteralExpr>(
                              arrow::MakeScalar((int64_t)res))
                        : std::make_shared<LiteralExpr>(arrow::MakeScalar(res));
      }
    }
    return std::make_shared<BinaryExpr>(left, right, op);
  }
  auto unary = std::dynamic_pointer_cast<UnaryExpr>(expr);
  if (unary && unary->op == OpType::NOT) {
    auto inner = std::dynamic_pointer_cast<UnaryExpr>(unary->operand);
    if (inner && inner->op == OpType::NOT) return simplify_expr(inner->operand);
    return std::make_shared<UnaryExpr>(simplify_expr(unary->operand),
                                       OpType::NOT);
  }
  return expr;
}

// ========================= OPTIMIZER =========================
static std::vector<Operation> optimize(const std::vector<Operation>& ops) {
  std::vector<Operation> result;
  for (size_t i = 0; i < ops.size(); i++) {
    auto op = ops[i];
    if (op.type == LazyOpType::WITH_COLUMN || op.type == LazyOpType::FILTER)
      op.expr = simplify_expr(op.expr);

    if (op.type == LazyOpType::FILTER) {
      size_t pos = result.size();
      while (pos > 0) {
        const auto& prev = result[pos - 1];
        if (prev.type == LazyOpType::GROUP_BY ||
            prev.type == LazyOpType::AGGREGATE ||
            prev.type == LazyOpType::JOIN || prev.type == LazyOpType::SORT ||
            prev.type == LazyOpType::HEAD)
          break;
        pos--;
      }
      result.insert(result.begin() + pos, op);
      continue;
    }

    if (op.type == LazyOpType::HEAD && !result.empty() &&
        result.back().type == LazyOpType::SORT) {
      result.back().n = op.n;
      continue;
    }

    if (op.type == LazyOpType::HEAD) {
      size_t pos = result.size();
      while (pos > 0) {
        const auto& prev = result[pos - 1];
        if (prev.type == LazyOpType::SORT ||
            prev.type == LazyOpType::GROUP_BY ||
            prev.type == LazyOpType::AGGREGATE ||
            prev.type == LazyOpType::JOIN ||
            prev.type == LazyOpType::WITH_COLUMN ||
            prev.type == LazyOpType::FILTER)
          break;
        pos--;
      }
      result.insert(result.begin() + pos, op);
      continue;
    }

    if (op.type == LazyOpType::SELECT) {
      size_t pos = result.size();
      while (pos > 0) {
        const auto& prev = result[pos - 1];
        if (prev.type == LazyOpType::GROUP_BY ||
            prev.type == LazyOpType::AGGREGATE ||
            prev.type == LazyOpType::JOIN ||
            prev.type == LazyOpType::WITH_COLUMN)
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

// ========================= COLLECT =========================
EagerDataFrame LazyDataFrame::collect() const {
  EagerDataFrame df = (source_type_ == SourceType::CSV)
                          ? read_csv(source_path_)
                          : read_parquet(source_path_);
  auto optimized = optimize(ops_);
  std::vector<std::string> group_keys;
  bool has_group = false;
  for (const auto& op : optimized) {
    switch (op.type) {
      case LazyOpType::FILTER:
        df = df.filter(ExprPtr(op.expr));
        break;
      case LazyOpType::SELECT:
        df = df.select(op.columns);
        break;
      case LazyOpType::WITH_COLUMN:
        df = df.with_column(op.column_name, ExprPtr(op.expr));
        break;
      case LazyOpType::SORT:
        if (op.n > 0)
          df = df.sort_head(op.columns, op.ascending, op.n);
        else
          df = df.sort(op.columns, op.ascending);
        break;
      case LazyOpType::HEAD:
        df = df.head(op.n);
        break;
      case LazyOpType::GROUP_BY:
        group_keys = op.columns;
        has_group = true;
        break;
      case LazyOpType::AGGREGATE:
        if (!has_group)
          throw std::runtime_error("aggregate() without group_by()");
        df = df.group_by(group_keys).aggregate(op.agg_map);
        has_group = false;
        break;
      case LazyOpType::JOIN: {
        auto right = op.join_df->collect();
        df = df.join(right, op.join_keys, op.join_how);
        break;
      }
    }
  }
  return df;
}

void LazyDataFrame::sink_csv(const std::string& path) const {
  collect().write_csv(path);
}
void LazyDataFrame::sink_parquet(const std::string& path) const {
  collect().write_parquet(path);
}

void LazyDataFrame::explain(const std::string& path) const {
  std::ofstream out(path);
  out << "digraph G {\n  rankdir=LR;\n  node0 [label=\"SCAN\"];\n";
  auto optimized = optimize(ops_);
  for (size_t i = 0; i < optimized.size(); i++) {
    std::string label;
    switch (optimized[i].type) {
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
    out << "  node" << i + 1 << " [label=\"" << label << "\"];\n";
    out << "  node" << i << " -> node" << i + 1 << ";\n";
  }
  out << "}\n";
  out.close();

  std::string png_path = path + ".png";
  std::string cmd =
      "dot -Tpng \"" + path + "\" -o \"" + png_path + "\" 2>/dev/null";
  system(cmd.c_str());
}

}  // namespace dataframelib