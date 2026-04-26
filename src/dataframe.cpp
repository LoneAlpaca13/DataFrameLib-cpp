#include "dataframe.h"

#include <algorithm>
#include <iostream>
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
  int cols = table->num_columns();

  for (int j = 0; j < cols; j++) {
    std::cout << table->field(j)->name() << "\t";
  }
  std::cout << "\n";

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      auto scalar = table->column(j)->chunk(0)->GetScalar(i).ValueOrDie();
      std::cout << scalar->ToString() << "\t";
    }
    std::cout << "\n";
  }
}

// ================= SELECT =================
EagerDataFrame EagerDataFrame::select(
    const std::vector<std::string>& columns) const {
  std::vector<std::shared_ptr<arrow::Field>> fields;
  std::vector<std::shared_ptr<arrow::ChunkedArray>> arrays;

  for (const auto& name : columns) {
    int idx = table->schema()->GetFieldIndex(name);
    if (idx == -1) throw std::runtime_error("Column not found: " + name);

    fields.push_back(table->field(idx));
    arrays.push_back(table->column(idx));
  }

  return EagerDataFrame(
      arrow::Table::Make(std::make_shared<arrow::Schema>(fields), arrays));
}

// ================= HEAD =================
EagerDataFrame EagerDataFrame::head(int n) const {
  return EagerDataFrame(table->Slice(0, std::min(n, (int)table->num_rows())));
}

// ================= FILTER =================
EagerDataFrame EagerDataFrame::filter(std::shared_ptr<Expr> predicate) const {
  auto mask = predicate->evaluate(table);

  std::vector<std::shared_ptr<arrow::Array>> new_arrays;

  for (int j = 0; j < table->num_columns(); j++) {
    auto chunk = table->column(j)->chunk(0);

    arrow::StringBuilder builder;

    for (int i = 0; i < chunk->length(); i++) {
      auto keep = std::dynamic_pointer_cast<arrow::BooleanScalar>(mask[i]);

      if (!keep || !keep->is_valid || !keep->value) continue;

      auto scalar = chunk->GetScalar(i).ValueOrDie();
      builder.Append(scalar->ToString());
    }

    std::shared_ptr<arrow::Array> arr;
    builder.Finish(&arr);
    new_arrays.push_back(arr);
  }

  return EagerDataFrame(arrow::Table::Make(table->schema(), new_arrays));
}

// ================= WITH COLUMN =================
EagerDataFrame EagerDataFrame::with_column(const std::string& name,
                                           std::shared_ptr<Expr> expr) const {
  auto values = expr->evaluate(table);

  arrow::StringBuilder builder;

  for (auto& v : values) {
    if (!v->is_valid) {
      builder.AppendNull();
    } else {
      builder.Append(v->ToString());
    }
  }

  std::shared_ptr<arrow::Array> arr;
  builder.Finish(&arr);

  auto chunked = std::make_shared<arrow::ChunkedArray>(arr);

  int idx = table->schema()->GetFieldIndex(name);

  std::shared_ptr<arrow::Table> new_table;

  if (idx == -1) {
    new_table = table
                    ->AddColumn(table->num_columns(),
                                arrow::field(name, arrow::utf8()), chunked)
                    .ValueOrDie();
  } else {
    new_table =
        table->SetColumn(idx, arrow::field(name, arrow::utf8()), chunked)
            .ValueOrDie();
  }

  return EagerDataFrame(new_table);
}

// ================= SORT =================
EagerDataFrame EagerDataFrame::sort(const std::string& column_name) const {
  int col_idx = table->schema()->GetFieldIndex(column_name);
  if (col_idx == -1) throw std::runtime_error("Column not found");

  auto chunk = table->column(col_idx)->chunk(0);

  std::vector<int> indices(chunk->length());
  std::iota(indices.begin(), indices.end(), 0);

  std::sort(indices.begin(), indices.end(), [&](int a, int b) {
    return chunk->GetScalar(a).ValueOrDie()->ToString() <
           chunk->GetScalar(b).ValueOrDie()->ToString();
  });

  std::vector<std::shared_ptr<arrow::ChunkedArray>> cols;

  for (int j = 0; j < table->num_columns(); j++) {
    auto ch = table->column(j)->chunk(0);

    arrow::StringBuilder builder;

    for (int idx : indices) {
      builder.Append(ch->GetScalar(idx).ValueOrDie()->ToString());
    }

    std::shared_ptr<arrow::Array> arr;
    builder.Finish(&arr);
    cols.push_back(std::make_shared<arrow::ChunkedArray>(arr));
  }

  return EagerDataFrame(arrow::Table::Make(table->schema(), cols));
}

// ================= GROUP BY =================
GroupedDataFrame EagerDataFrame::group_by(
    const std::vector<std::string>& column_names) const {
  if (column_names.empty()) throw std::runtime_error("group_by needs column");

  int idx = table->schema()->GetFieldIndex(column_names[0]);
  if (idx == -1) throw std::runtime_error("Column not found");

  auto chunk = table->column(idx)->chunk(0);

  std::map<std::string, std::vector<int>> groups;

  for (int i = 0; i < chunk->length(); i++) {
    auto scalar = chunk->GetScalar(i).ValueOrDie();

    if (!scalar->is_valid) continue;  // ✅ skip null keys

    std::string key = scalar->ToString();
    groups[key].push_back(i);
  }

  return GroupedDataFrame(table, groups);
}

// ================= PRINT GROUPS =================
void GroupedDataFrame::printGroups() const {
  for (const auto& [k, v] : groups) {
    std::cout << k << " : ";
    for (int i : v) std::cout << i << " ";
    std::cout << "\n";
  }
}

// ================= AGGREGATE =================
EagerDataFrame GroupedDataFrame::aggregate(const std::string& column_name,
                                           const std::string& op) const {
  int idx = table->schema()->GetFieldIndex(column_name);
  if (idx == -1) throw std::runtime_error("Column not found");

  auto chunk = table->column(idx)->chunk(0);

  arrow::StringBuilder key_builder;
  arrow::Int64Builder val_builder;

  for (auto& [k, indices] : groups) {
    int64_t result = 0;

    if (op == "count") {
      result = indices.size();
    }

    else if (op == "sum") {
      for (int i : indices) {
        auto scalar = chunk->GetScalar(i).ValueOrDie();

        if (!scalar->is_valid) continue;

        auto val = std::dynamic_pointer_cast<arrow::Int64Scalar>(scalar);
        result += val->value;
      }
    }

    else if (op == "mean") {
      int64_t sum = 0;
      int count = 0;

      for (int i : indices) {
        auto scalar = chunk->GetScalar(i).ValueOrDie();

        if (!scalar->is_valid) continue;

        auto val = std::dynamic_pointer_cast<arrow::Int64Scalar>(scalar);
        sum += val->value;
        count++;
      }

      if (count > 0) result = sum / count;
    }

    else {
      throw std::runtime_error("Unsupported aggregation");
    }

    key_builder.Append(k);
    val_builder.Append(result);
  }

  std::shared_ptr<arrow::Array> keys, vals;
  key_builder.Finish(&keys);
  val_builder.Finish(&vals);

  auto schema =
      arrow::schema({arrow::field("group_key", arrow::utf8()),
                     arrow::field(column_name + "_" + op, arrow::int64())});

  return EagerDataFrame(arrow::Table::Make(
      schema, {std::make_shared<arrow::ChunkedArray>(keys),
               std::make_shared<arrow::ChunkedArray>(vals)}));
}

// ================= JOIN =================
EagerDataFrame EagerDataFrame::join(
    const EagerDataFrame& other, const std::vector<std::string>& column_names,
    const std::string& how) const {
  if (how != "inner") throw std::runtime_error("Only inner join supported");

  if (column_names.empty()) throw std::runtime_error("join needs column");

  std::string col = column_names[0];

  int lidx = table->schema()->GetFieldIndex(col);
  int ridx = other.getTable()->schema()->GetFieldIndex(col);

  auto lcol = table->column(lidx)->chunk(0);
  auto rcol = other.getTable()->column(ridx)->chunk(0);

  std::unordered_map<std::string, std::vector<int>> map;

  for (int i = 0; i < rcol->length(); i++) {
    map[rcol->GetScalar(i).ValueOrDie()->ToString()].push_back(i);
  }

  std::vector<std::shared_ptr<arrow::ChunkedArray>> cols;

  std::vector<arrow::StringBuilder> builders(
      table->num_columns() + other.getTable()->num_columns() - 1);

  for (int i = 0; i < lcol->length(); i++) {
    std::string key = lcol->GetScalar(i).ValueOrDie()->ToString();

    if (!map.count(key)) continue;

    for (int r : map[key]) {
      int pos = 0;

      for (int j = 0; j < table->num_columns(); j++) {
        builders[pos++].Append(
            table->column(j)->chunk(0)->GetScalar(i).ValueOrDie()->ToString());
      }

      for (int j = 0; j < other.getTable()->num_columns(); j++) {
        if (j == ridx) continue;
        builders[pos++].Append(other.getTable()
                                   ->column(j)
                                   ->chunk(0)
                                   ->GetScalar(r)
                                   .ValueOrDie()
                                   ->ToString());
      }
    }
  }

  std::vector<std::shared_ptr<arrow::Field>> fields;

  for (int j = 0; j < table->num_columns(); j++)
    fields.push_back(table->field(j));

  for (int j = 0; j < other.getTable()->num_columns(); j++)
    if (j != ridx) fields.push_back(other.getTable()->field(j));

  for (auto& b : builders) {
    std::shared_ptr<arrow::Array> arr;
    b.Finish(&arr);
    cols.push_back(std::make_shared<arrow::ChunkedArray>(arr));
  }

  return EagerDataFrame(
      arrow::Table::Make(std::make_shared<arrow::Schema>(fields), cols));
}

}  // namespace dataframelib