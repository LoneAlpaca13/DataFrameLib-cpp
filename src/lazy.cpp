#include "lazy.h"

#include <fstream>

#include "dataframelib/dataframelib.h"

namespace dataframelib {

// ================= HELPERS =================

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

// ================= OPTIMIZER =================

std::vector<Operation> optimize(const std::vector<Operation>& ops) {
  std::vector<Operation> result;

  for (auto op : ops) {
    // =========================
    // 1. FILTER PUSHDOWN
    // =========================
    if (op.type == LazyOpType::FILTER) {
      std::string filter_col = getFilterColumn(op.expr);

      size_t pos = result.size();

      while (pos > 0) {
        const auto& prev = result[pos - 1];

        bool can_swap = true;

        // cannot cross column creation if depends
        if (prev.type == LazyOpType::WITH_COLUMN &&
            prev.column_name == filter_col) {
          can_swap = false;
        }

        // cannot cross grouping
        if (prev.type == LazyOpType::GROUP_BY ||
            prev.type == LazyOpType::AGGREGATE) {
          can_swap = false;
        }

        if (!can_swap) break;
        pos--;
      }

      result.insert(result.begin() + pos, op);
      continue;
    }

    // =========================
    // 2. PROJECTION PUSHDOWN
    // =========================
    if (op.type == LazyOpType::SELECT) {
      // remove redundant selects
      if (!result.empty() && result.back().type == LazyOpType::SELECT) {
        result.back() = op;  // overwrite previous select
      } else {
        result.push_back(op);
      }
      continue;
    }

    // =========================
    // 3. EXPRESSION SIMPLIFICATION
    // =========================
    if (op.type == LazyOpType::WITH_COLUMN) {
      auto bin = std::dynamic_pointer_cast<BinaryExpr>(op.expr);

      if (bin) {
        auto left = bin->getLeft();
        auto right = bin->getRight();

        // x + 0 → x
        if (bin->getOp() == OpType::ADD) {
          // x + 0 → x
          if (bin->getOp() == OpType::ADD) {
            auto lit = std::dynamic_pointer_cast<LiteralExpr>(right);
            if (lit) {
              auto val = std::dynamic_pointer_cast<arrow::Int64Scalar>(
                  lit->getValue());
              if (val && val->value == 0) op.expr = left;
            }
          }

          // x * 1 → x
          if (bin->getOp() == OpType::MUL) {
            auto lit = std::dynamic_pointer_cast<LiteralExpr>(right);
            if (lit) {
              auto val = std::dynamic_pointer_cast<arrow::Int64Scalar>(
                  lit->getValue());
              if (val && val->value == 1) op.expr = left;
            }
          }
        }

        // x * 1 → x
        if (bin->getOp() == OpType::MUL) {
          if (auto lit = std::dynamic_pointer_cast<LiteralExpr>(right)) {
            op.expr = left;
          }
        }
      }
    }

    // default
    result.push_back(op);
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

GroupedDataFrame EagerDataFrame::group_by(
    const std::vector<std::string>& column_names) const {
  std::map<std::string, std::vector<int>> groups;

  for (int i = 0; i < table->num_rows(); i++) {
    std::string key = "";

    for (const auto& col_name : column_names) {
      int idx = table->schema()->GetFieldIndex(col_name);
      if (idx == -1) throw std::runtime_error("Column not found");

      auto scalar = table->column(idx)->chunk(0)->GetScalar(i).ValueOrDie();

      key += (scalar->is_valid ? scalar->ToString() : "null") + "|";
    }

    groups[key].push_back(i);
  }

  return GroupedDataFrame(table, groups);
}

EagerDataFrame GroupedDataFrame::aggregate(
    const std::vector<std::pair<std::string, std::string>>& agg_map) const {
  std::vector<std::shared_ptr<arrow::Array>> result_arrays;
  std::vector<std::shared_ptr<arrow::Field>> fields;

  // group key column
  arrow::StringBuilder key_builder;

  for (const auto& [key, _] : groups) {
    key_builder.Append(key);
  }

  std::shared_ptr<arrow::Array> key_array;
  key_builder.Finish(&key_array);

  result_arrays.push_back(key_array);
  fields.push_back(arrow::field("group_key", arrow::utf8()));

  // each aggregation
  for (const auto& [col_name, op] : agg_map) {
    int idx = table->schema()->GetFieldIndex(col_name);
    if (idx == -1) throw std::runtime_error("Column not found");

    auto chunk = table->column(idx)->chunk(0);

    arrow::Int64Builder builder;

    for (const auto& [_, rows] : groups) {
      int64_t sum = 0;
      int count = 0;
      int64_t minv = 0, maxv = 0;
      bool found = false;

      for (int i : rows) {
        auto s = chunk->GetScalar(i).ValueOrDie();
        if (!s->is_valid) continue;

        auto val = std::dynamic_pointer_cast<arrow::Int64Scalar>(s);
        if (!val) continue;

        int64_t v = val->value;

        if (!found) {
          minv = maxv = v;
          found = true;
        }

        sum += v;
        count++;

        minv = std::min(minv, v);
        maxv = std::max(maxv, v);
      }

      if (!found) {
        builder.AppendNull();
        continue;
      }

      if (op == "sum")
        builder.Append(sum);
      else if (op == "mean")
        builder.Append(count ? sum / count : 0);
      else if (op == "count")
        builder.Append(count);
      else if (op == "min")
        builder.Append(minv);
      else if (op == "max")
        builder.Append(maxv);
      else
        throw std::runtime_error("Unknown aggregation");
    }

    std::shared_ptr<arrow::Array> arr;
    builder.Finish(&arr);

    result_arrays.push_back(arr);
    fields.push_back(arrow::field(col_name + "_" + op, arrow::int64()));
  }

  return EagerDataFrame(arrow::Table::Make(
      std::make_shared<arrow::Schema>(fields), result_arrays));
}

LazyDataFrame LazyDataFrame::sort(const std::vector<std::string>& cols,
                                  bool /*asc*/) const {
  LazyDataFrame df = *this;

  Operation op;
  op.type = LazyOpType::SORT;
  op.columns = cols;

  df.ops.push_back(op);
  return df;
}

LazyDataFrame LazyDataFrame::head(int n) const {
  LazyDataFrame df = *this;

  Operation op;
  op.type = LazyOpType::HEAD;
  op.n = n;

  df.ops.push_back(op);
  return df;
}

// ================= EXPLAIN =================

void LazyDataFrame::explain(const std::string& path) const {
  std::ofstream out(path);

  out << "=== Logical Plan ===\n";

  for (const auto& op : ops) {
    switch (op.type) {
      case LazyOpType::FILTER:
        out << "FILTER\n";
        break;

      case LazyOpType::SELECT:
        out << "SELECT [ ";
        for (auto& c : op.columns) out << c << " ";
        out << "]\n";
        break;

      case LazyOpType::WITH_COLUMN:
        out << "WITH_COLUMN " << op.column_name << "\n";
        break;

      case LazyOpType::GROUP_BY:
        out << "GROUP_BY [ ";
        for (auto& c : op.columns) out << c << " ";
        out << "]\n";
        break;

      case LazyOpType::AGGREGATE:
        out << "AGGREGATE\n";
        break;

      case LazyOpType::SORT:
        out << "SORT " << op.columns[0] << "\n";
        break;

      case LazyOpType::HEAD:
        out << "HEAD " << op.n << "\n";
        break;
    }
  }

  out.close();
}

// ================= COLLECT =================

EagerDataFrame LazyDataFrame::collect() const {
  auto df = dataframelib::read_csv(csv_path);

  std::vector<std::string> group_keys;
  bool has_group = false;

  auto optimized_ops = optimize(ops);

  for (const auto& op : optimized_ops) {
    if (op.type == LazyOpType::FILTER) {
      df = df.filter(op.expr);
    }

    else if (op.type == LazyOpType::SELECT) {
      std::vector<std::shared_ptr<Expr>> exprs;
      for (const auto& c : op.columns) {
        exprs.push_back(col(c));
      }

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
  }

  return df;
}

EagerDataFrame EagerDataFrame::join(const EagerDataFrame& other,
                                    const std::vector<std::string>& keys,
                                    const std::string& how) const {
  if (keys.empty()) throw std::runtime_error("Join keys required");

  int left_idx = table->schema()->GetFieldIndex(keys[0]);
  int right_idx = other.table->schema()->GetFieldIndex(keys[0]);

  if (left_idx == -1 || right_idx == -1)
    throw std::runtime_error("Join column not found");

  auto left_col = table->column(left_idx)->chunk(0);
  auto right_col = other.table->column(right_idx)->chunk(0);

  std::vector<std::vector<std::string>> rows;

  for (int i = 0; i < left_col->length(); i++) {
    auto l = left_col->GetScalar(i).ValueOrDie();

    for (int j = 0; j < right_col->length(); j++) {
      auto r = right_col->GetScalar(j).ValueOrDie();

      if (l->ToString() == r->ToString()) {
        std::vector<std::string> row;

        for (int c = 0; c < table->num_columns(); c++)
          row.push_back(table->column(c)
                            ->chunk(0)
                            ->GetScalar(i)
                            .ValueOrDie()
                            ->ToString());
        for (int c = 0; c < other.table->num_columns(); c++)
          row.push_back(other.table->column(c)
                            ->chunk(0)
                            ->GetScalar(j)
                            .ValueOrDie()
                            ->ToString());
        rows.push_back(row);
      }
    }
  }

  // build result as strings (simple but works)
  std::vector<std::shared_ptr<arrow::Array>> arrays;

  int total_cols = table->num_columns() + other.table->num_columns();

  for (int c = 0; c < total_cols; c++) {
    arrow::StringBuilder builder;
    for (auto& row : rows) builder.Append(row[c]);

    std::shared_ptr<arrow::Array> arr;
    builder.Finish(&arr);
    arrays.push_back(arr);
  }

  std::vector<std::shared_ptr<arrow::Field>> fields;

  for (int c = 0; c < total_cols; c++)
    fields.push_back(arrow::field("col" + std::to_string(c), arrow::utf8()));

  return EagerDataFrame(
      arrow::Table::Make(std::make_shared<arrow::Schema>(fields), arrays));
}

}  // namespace dataframelib