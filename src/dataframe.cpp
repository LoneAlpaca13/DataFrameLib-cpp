#include "dataframe.h"

#include <arrow/csv/writer.h>
#include <arrow/io/api.h>
#include <parquet/arrow/writer.h>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <unordered_map>

namespace dataframelib {

// ================= CONSTRUCTOR =================

EagerDataFrame::EagerDataFrame(std::shared_ptr<arrow::Table> t) : table(t) {}

std::shared_ptr<arrow::Table> EagerDataFrame::getTable() const { return table; }

// ================= PRINT =================

void EagerDataFrame::printSchema() const {
  std::cout << table->schema()->ToString() << std::endl;
}

void EagerDataFrame::printHead(int n) const {
  int rows = std::min(n, (int)table->num_rows());

  for (int j = 0; j < table->num_columns(); j++)
    std::cout << table->field(j)->name() << "\t";
  std::cout << "\n";

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < table->num_columns(); j++) {
      auto scalar = table->column(j)->chunk(0)->GetScalar(i).ValueOrDie();
      std::cout << scalar->ToString() << "\t";
    }
    std::cout << "\n";
  }
}
EagerDataFrame EagerDataFrame::head(int n) const {
  return EagerDataFrame(table->Slice(0, n));
}
// ================= SELECT (EXPR BASED) =================

EagerDataFrame EagerDataFrame::select(
    const std::vector<std::shared_ptr<Expr>>& exprs) const {
  std::vector<std::shared_ptr<arrow::Field>> fields;
  std::vector<std::shared_ptr<arrow::ChunkedArray>> arrays;

  for (size_t i = 0; i < exprs.size(); i++) {
    auto values = exprs[i]->evaluate(table);

    auto first = values[0];

    std::shared_ptr<arrow::Array> arr;

    // ===== INT =====
    if (std::dynamic_pointer_cast<arrow::Int64Scalar>(first)) {
      arrow::Int64Builder builder;
      for (auto& v : values) {
        if (!v->is_valid)
          builder.AppendNull();
        else
          builder.Append(
              std::dynamic_pointer_cast<arrow::Int64Scalar>(v)->value);
      }
      builder.Finish(&arr);
      auto alias_expr = std::dynamic_pointer_cast<AliasExpr>(exprs[i]);

      std::string name =
          alias_expr ? alias_expr->getAlias() : "col" + std::to_string(i);

      fields.push_back(arrow::field(name, arr->type()));
    }

    // ===== BOOL =====
    else if (std::dynamic_pointer_cast<arrow::BooleanScalar>(first)) {
      arrow::BooleanBuilder builder;
      for (auto& v : values) {
        if (!v->is_valid)
          builder.AppendNull();
        else
          builder.Append(
              std::dynamic_pointer_cast<arrow::BooleanScalar>(v)->value);
      }
      builder.Finish(&arr);
      fields.push_back(
          arrow::field("col" + std::to_string(i), arrow::boolean()));
    }

    // ===== STRING =====
    else {
      arrow::StringBuilder builder;
      for (auto& v : values) {
        if (!v->is_valid)
          builder.AppendNull();
        else
          builder.Append(v->ToString());
      }
      builder.Finish(&arr);
      fields.push_back(arrow::field("col" + std::to_string(i), arrow::utf8()));
    }

    arrays.push_back(std::make_shared<arrow::ChunkedArray>(arr));
  }

  return EagerDataFrame(
      arrow::Table::Make(std::make_shared<arrow::Schema>(fields), arrays));
}

// ================= FILTER =================

EagerDataFrame EagerDataFrame::filter(std::shared_ptr<Expr> predicate) const {
  auto mask = predicate->evaluate(table);

  std::vector<int> selected;

  for (int i = 0; i < (int)mask.size(); i++) {
    auto keep = std::dynamic_pointer_cast<arrow::BooleanScalar>(mask[i]);
    if (keep && keep->is_valid && keep->value) selected.push_back(i);
  }

  std::vector<std::shared_ptr<arrow::ChunkedArray>> new_cols;

  for (int j = 0; j < table->num_columns(); j++) {
    auto col = table->column(j)->chunk(0);

    std::shared_ptr<arrow::Array> arr;

    if (col->type_id() == arrow::Type::INT64) {
      arrow::Int64Builder builder;
      for (int i : selected) {
        auto v = std::dynamic_pointer_cast<arrow::Int64Scalar>(
            col->GetScalar(i).ValueOrDie());
        if (!v->is_valid)
          builder.AppendNull();
        else
          builder.Append(v->value);
      }
      builder.Finish(&arr);
    }

    else if (col->type_id() == arrow::Type::BOOL) {
      arrow::BooleanBuilder builder;
      for (int i : selected) {
        auto v = std::dynamic_pointer_cast<arrow::BooleanScalar>(
            col->GetScalar(i).ValueOrDie());
        if (!v->is_valid)
          builder.AppendNull();
        else
          builder.Append(v->value);
      }
      builder.Finish(&arr);
    }

    else {
      arrow::StringBuilder builder;
      for (int i : selected) {
        auto v = col->GetScalar(i).ValueOrDie();
        if (!v->is_valid)
          builder.AppendNull();
        else
          builder.Append(v->ToString());
      }
      builder.Finish(&arr);
    }

    new_cols.push_back(std::make_shared<arrow::ChunkedArray>(arr));
  }

  return EagerDataFrame(arrow::Table::Make(table->schema(), new_cols));
}

// ================= WITH COLUMN =================

EagerDataFrame EagerDataFrame::with_column(const std::string& name,
                                           std::shared_ptr<Expr> expr) const {
  auto values = expr->evaluate(table);
  auto first = values[0];

  std::shared_ptr<arrow::Array> arr;

  if (std::dynamic_pointer_cast<arrow::Int64Scalar>(first)) {
    arrow::Int64Builder builder;
    for (auto& v : values) {
      if (!v->is_valid)
        builder.AppendNull();
      else
        builder.Append(std::dynamic_pointer_cast<arrow::Int64Scalar>(v)->value);
    }
    builder.Finish(&arr);
  }

  else if (std::dynamic_pointer_cast<arrow::BooleanScalar>(first)) {
    arrow::BooleanBuilder builder;
    for (auto& v : values) {
      if (!v->is_valid)
        builder.AppendNull();
      else
        builder.Append(
            std::dynamic_pointer_cast<arrow::BooleanScalar>(v)->value);
    }
    builder.Finish(&arr);
  }

  else {
    arrow::StringBuilder builder;
    for (auto& v : values) {
      if (!v->is_valid)
        builder.AppendNull();
      else
        builder.Append(v->ToString());
    }
    builder.Finish(&arr);
  }

  auto chunked = std::make_shared<arrow::ChunkedArray>(arr);

  int idx = table->schema()->GetFieldIndex(name);

  if (idx == -1) {
    return EagerDataFrame(table
                              ->AddColumn(table->num_columns(),
                                          arrow::field(name, arr->type()),
                                          chunked)
                              .ValueOrDie());
  }

  return EagerDataFrame(
      table->SetColumn(idx, arrow::field(name, arr->type()), chunked)
          .ValueOrDie());
}

// ================= SORT =================

EagerDataFrame EagerDataFrame::sort(const std::string& column_name,
                                    bool asc) const {
  int idx = table->schema()->GetFieldIndex(column_name);
  if (idx == -1) throw std::runtime_error("Column not found");

  auto col = table->column(idx)->chunk(0);

  std::vector<int> indices(col->length());
  std::iota(indices.begin(), indices.end(), 0);

  std::sort(indices.begin(), indices.end(), [&](int a, int b) {
    auto A = col->GetScalar(a).ValueOrDie();
    auto B = col->GetScalar(b).ValueOrDie();

    if (!A->is_valid) return false;
    if (!B->is_valid) return true;

    if (auto ai = std::dynamic_pointer_cast<arrow::Int64Scalar>(A)) {
      auto bi = std::dynamic_pointer_cast<arrow::Int64Scalar>(B);
      return asc ? ai->value < bi->value : ai->value > bi->value;
    }

    return asc ? A->ToString() < B->ToString() : A->ToString() > B->ToString();
  });

  std::vector<std::shared_ptr<arrow::ChunkedArray>> new_cols;

  for (int j = 0; j < table->num_columns(); j++) {
    auto c = table->column(j)->chunk(0);

    std::shared_ptr<arrow::Array> arr;

    if (c->type_id() == arrow::Type::INT64) {
      arrow::Int64Builder builder;
      for (int i : indices) {
        auto v = std::dynamic_pointer_cast<arrow::Int64Scalar>(
            c->GetScalar(i).ValueOrDie());
        if (!v->is_valid)
          builder.AppendNull();
        else
          builder.Append(v->value);
      }
      builder.Finish(&arr);
    } else {
      arrow::StringBuilder builder;
      for (int i : indices) {
        auto v = c->GetScalar(i).ValueOrDie();
        if (!v->is_valid)
          builder.AppendNull();
        else
          builder.Append(v->ToString());
      }
      builder.Finish(&arr);
    }

    new_cols.push_back(std::make_shared<arrow::ChunkedArray>(arr));
  }

  return EagerDataFrame(arrow::Table::Make(table->schema(), new_cols));
}

}  // namespace dataframelib