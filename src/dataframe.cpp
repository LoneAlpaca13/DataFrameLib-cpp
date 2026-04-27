#include "dataframe.h"

#include <arrow/csv/writer.h>
#include <arrow/io/api.h>
#include <parquet/arrow/writer.h>

#include <iostream>
#include <stdexcept>

// ================= ERROR HANDLING =================
#define CHECK_ARROW(x)                                     \
  do {                                                     \
    arrow::Status _s = (x);                                \
    if (!_s.ok()) throw std::runtime_error(_s.ToString()); \
  } while (0)

namespace dataframelib {

// ================= HELPER =================
std::shared_ptr<arrow::Array> scalars_to_array(
    const std::vector<std::shared_ptr<arrow::Scalar>>& scalars) {
  arrow::StringBuilder builder;

  for (const auto& s : scalars) {
    if (!s || !s->is_valid)
      CHECK_ARROW(builder.AppendNull());
    else
      CHECK_ARROW(builder.Append(s->ToString()));
  }

  std::shared_ptr<arrow::Array> arr;
  CHECK_ARROW(builder.Finish(&arr));
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

    fields.push_back(arrow::field("col" + std::to_string(i), arr->type()));
    arrays.push_back(arr);
  }

  return EagerDataFrame(
      arrow::Table::Make(std::make_shared<arrow::Schema>(fields), arrays));
}

// ================= FILTER =================
EagerDataFrame EagerDataFrame::filter(std::shared_ptr<Expr> predicate) const {
  auto mask = predicate->evaluate(table);

  std::vector<std::shared_ptr<arrow::Array>> new_cols;

  for (int c = 0; c < table->num_columns(); c++) {
    auto arr = table->column(c)->chunk(0);

    arrow::StringBuilder builder;

    for (int i = 0; i < table->num_rows(); i++) {
      auto m = std::dynamic_pointer_cast<arrow::BooleanScalar>(mask[i]);

      if (m && m->value) {
        CHECK_ARROW(builder.Append(arr->GetScalar(i).ValueOrDie()->ToString()));
      }
    }

    std::shared_ptr<arrow::Array> out;
    CHECK_ARROW(builder.Finish(&out));
    new_cols.push_back(out);
  }

  return EagerDataFrame(arrow::Table::Make(table->schema(), new_cols));
}

// ================= WITH COLUMN =================
EagerDataFrame EagerDataFrame::with_column(const std::string& name,
                                           std::shared_ptr<Expr> expr) const {
  auto scalars = expr->evaluate(table);
  auto arr = scalars_to_array(scalars);

  auto chunked = std::make_shared<arrow::ChunkedArray>(arr);

  auto new_table = table
                       ->AddColumn(table->num_columns(),
                                   arrow::field(name, arr->type()), chunked)
                       .ValueOrDie();

  return EagerDataFrame(new_table);
}

// ================= SORT =================
EagerDataFrame EagerDataFrame::sort(const std::string& column_name) const {
  int idx = table->schema()->GetFieldIndex(column_name);
  auto arr = table->column(idx)->chunk(0);

  std::vector<int> indices(table->num_rows());
  for (int i = 0; i < indices.size(); i++) indices[i] = i;

  std::sort(indices.begin(), indices.end(), [&](int a, int b) {
    return arr->GetScalar(a).ValueOrDie()->ToString() <
           arr->GetScalar(b).ValueOrDie()->ToString();
  });

  std::vector<std::shared_ptr<arrow::Array>> new_cols;

  for (int c = 0; c < table->num_columns(); c++) {
    auto col = table->column(c)->chunk(0);
    arrow::StringBuilder builder;

    for (int i : indices)
      CHECK_ARROW(builder.Append(col->GetScalar(i).ValueOrDie()->ToString()));

    std::shared_ptr<arrow::Array> out;
    CHECK_ARROW(builder.Finish(&out));
    new_cols.push_back(out);
  }

  return EagerDataFrame(arrow::Table::Make(table->schema(), new_cols));
}

// ================= HEAD =================
EagerDataFrame EagerDataFrame::head(int n) const {
  return EagerDataFrame(table->Slice(0, n));
}

// ================= GROUP BY =================
GroupedDataFrame EagerDataFrame::group_by(
    const std::vector<std::string>& column_names) const {
  std::map<std::string, std::vector<int>> groups;

  for (int i = 0; i < table->num_rows(); i++) {
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
    CHECK_ARROW(builder.Append(key));
  }

  std::shared_ptr<arrow::Array> arr;
  CHECK_ARROW(builder.Finish(&arr));

  auto schema = std::make_shared<arrow::Schema>(
      std::vector<std::shared_ptr<arrow::Field>>{
          arrow::field("group", arrow::utf8())});

  return EagerDataFrame(arrow::Table::Make(schema, {arr}));
}

// ================= JOIN =================
EagerDataFrame EagerDataFrame::join(
    const EagerDataFrame& other, const std::vector<std::string>& column_names,
    const std::string& how) const {
  if (how != "inner" && how != "left")
    throw std::runtime_error("Unsupported join");

  return *this;  // minimal safe stub
}

// ================= WRITE =================
void EagerDataFrame::write_csv(const std::string& path) const {
  auto out = arrow::io::FileOutputStream::Open(path).ValueOrDie();

  CHECK_ARROW(arrow::csv::WriteCSV(*table, arrow::csv::WriteOptions::Defaults(),
                                   out.get()));
}

void EagerDataFrame::write_parquet(const std::string& path) const {
  auto out = arrow::io::FileOutputStream::Open(path).ValueOrDie();

  auto writer = parquet::arrow::FileWriter::Open(
                    *table->schema(), arrow::default_memory_pool(), out)
                    .ValueOrDie();

  CHECK_ARROW(writer->WriteTable(*table, table->num_rows()));
  CHECK_ARROW(writer->Close());
}

}  // namespace dataframelib