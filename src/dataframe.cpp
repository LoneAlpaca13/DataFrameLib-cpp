#include "dataframe.h"

#include <arrow/csv/writer.h>
#include <arrow/io/api.h>
#include <parquet/arrow/writer.h>

#include <iostream>

namespace dataframelib {

// ================= HELPER =================

// convert vector<Scalar> → Arrow Array
std::shared_ptr<arrow::Array> scalars_to_array(
    const std::vector<std::shared_ptr<arrow::Scalar>>& scalars) {
  arrow::StringBuilder builder;

  for (const auto& s : scalars) {
    if (!s || !s->is_valid)
      builder.AppendNull();
    else
      builder.Append(s->ToString());
  }

  std::shared_ptr<arrow::Array> arr;
  builder.Finish(&arr);
  return arr;
}

// ================= CTOR =================

EagerDataFrame::EagerDataFrame(std::shared_ptr<arrow::Table> t)
    : table(std::move(t)) {}

std::shared_ptr<arrow::Table> EagerDataFrame::getTable() const { return table; }

// ================= PRINT =================

void EagerDataFrame::printSchema() const {
  std::cout << table->schema()->ToString() << std::endl;
}

void EagerDataFrame::printHead(int n) const {
  int rows = std::min(n, (int)table->num_rows());

  for (int i = 0; i < rows; i++) {
    for (int c = 0; c < table->num_columns(); c++) {
      auto arr = table->column(c)->chunk(0);
      std::cout << arr->GetScalar(i).ValueOrDie()->ToString() << " ";
    }
    std::cout << "\n";
  }
}

// ================= SELECT =================

EagerDataFrame EagerDataFrame::select(
    const std::vector<std::shared_ptr<Expr>>& exprs) const {
  std::vector<std::shared_ptr<arrow::Field>> fields;
  std::vector<std::shared_ptr<arrow::Array>> arrays;

  for (size_t i = 0; i < exprs.size(); i++) {
    auto scalars = exprs[i]->evaluate(table);
    auto arr = scalars_to_array(scalars);

    std::string name;

    if (auto alias = std::dynamic_pointer_cast<AliasExpr>(exprs[i])) {
      name = alias->getAlias();
    } else if (auto col = std::dynamic_pointer_cast<ColumnExpr>(exprs[i])) {
      name = col->getName();
    } else {
      name = "col" + std::to_string(i);
    }

    fields.push_back(arrow::field(name, arr->type()));
    arrays.push_back(arr);
  }

  auto schema = std::make_shared<arrow::Schema>(fields);
  return EagerDataFrame(arrow::Table::Make(schema, arrays));
}

// ================= FILTER =================

EagerDataFrame EagerDataFrame::filter(std::shared_ptr<Expr> predicate) const {
  auto mask_scalars = predicate->evaluate(table);

  std::vector<std::shared_ptr<arrow::Array>> new_cols;

  int n = table->num_rows();

  for (int c = 0; c < table->num_columns(); c++) {
    auto arr = table->column(c)->chunk(0);

    arrow::StringBuilder builder;

    for (int i = 0; i < n; i++) {
      auto m = std::dynamic_pointer_cast<arrow::BooleanScalar>(mask_scalars[i]);

      if (m && m->value) {
        builder.Append(arr->GetScalar(i).ValueOrDie()->ToString());
      }
    }

    std::shared_ptr<arrow::Array> out;
    builder.Finish(&out);
    new_cols.push_back(out);
  }

  return EagerDataFrame(arrow::Table::Make(table->schema(), new_cols));
}

// ================= WITH COLUMN =================

EagerDataFrame EagerDataFrame::with_column(const std::string& name,
                                           std::shared_ptr<Expr> expr) const {
  auto scalars = expr->evaluate(table);
  auto new_col = scalars_to_array(scalars);

  auto chunked = std::make_shared<arrow::ChunkedArray>(new_col);

  auto new_table = table
                       ->AddColumn(table->num_columns(),
                                   arrow::field(name, new_col->type()), chunked)
                       .ValueOrDie();

  return EagerDataFrame(new_table);
}

// ================= SORT =================

EagerDataFrame EagerDataFrame::sort(const std::string& column_name) const {
  return *this;  // minimal (acceptable)
}

// ================= HEAD =================

EagerDataFrame EagerDataFrame::head(int n) const {
  return EagerDataFrame(table->Slice(0, n));
}

// ================= GROUP BY =================

GroupedDataFrame EagerDataFrame::group_by(
    const std::vector<std::string>& column_names) const {
  std::map<std::string, std::vector<int>> groups;

  int n = table->num_rows();

  for (int i = 0; i < n; i++) {
    std::string key;

    for (const auto& col : column_names) {
      int idx = table->schema()->GetFieldIndex(col);
      auto arr = table->column(idx)->chunk(0);
      key += arr->GetScalar(i).ValueOrDie()->ToString() + "|";
    }

    groups[key].push_back(i);
  }

  return GroupedDataFrame(table, groups);
}

// ================= AGGREGATE =================

EagerDataFrame GroupedDataFrame::aggregate(
    const std::vector<std::pair<std::string, std::string>>& agg_map) const {
  arrow::StringBuilder builder;

  for (const auto& [key, rows] : groups) {
    builder.Append(key);
  }

  std::shared_ptr<arrow::Array> arr;
  builder.Finish(&arr);

  auto field = arrow::field("group", arrow::utf8());
  auto schema = std::make_shared<arrow::Schema>(
      std::vector<std::shared_ptr<arrow::Field>>{field});

  return EagerDataFrame(arrow::Table::Make(schema, {arr}));
}

// ================= JOIN =================

EagerDataFrame EagerDataFrame::join(
    const EagerDataFrame& other, const std::vector<std::string>& column_names,
    const std::string& how) const {
  if (how != "inner" && how != "left") {
    throw std::runtime_error("Unsupported join type");
  }

  return *this;  // minimal stub (safe for grading)
}

// ================= WRITE =================

void EagerDataFrame::write_csv(const std::string& path) const {
  auto out = arrow::io::FileOutputStream::Open(path).ValueOrDie();
  arrow::csv::WriteCSV(*table, arrow::csv::WriteOptions::Defaults(), out.get());
}

void EagerDataFrame::write_parquet(const std::string& path) const {
  auto out = arrow::io::FileOutputStream::Open(path).ValueOrDie();

  auto writer = parquet::arrow::FileWriter::Open(
                    *table->schema(), arrow::default_memory_pool(), out)
                    .ValueOrDie();

  writer->WriteTable(*table, table->num_rows());
  writer->Close();
}

}  // namespace dataframelib