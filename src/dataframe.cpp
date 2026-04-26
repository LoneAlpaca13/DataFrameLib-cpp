#include "dataframe.h"

#include <iostream>

EagerDataFrame::EagerDataFrame(std::shared_ptr<arrow::Table> t) : table(t) {}

std::shared_ptr<arrow::Table> EagerDataFrame::getTable() const { return table; }

void EagerDataFrame::printSchema() const {
  std::cout << table->schema()->ToString() << std::endl;
}

void EagerDataFrame::printHead(int n) const {
  int rows = std::min(n, (int)table->num_rows());
  int cols = table->num_columns();

  // header
  for (int j = 0; j < cols; j++) {
    std::cout << table->field(j)->name() << "\t";
  }
  std::cout << "\n";

  // rows
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      auto col = table->column(j);
      auto chunk = col->chunk(0);  // assume single chunk for now

      auto scalar = chunk->GetScalar(i).ValueOrDie();
      std::cout << scalar->ToString() << "\t";
    }
    std::cout << "\n";
  }
}

EagerDataFrame EagerDataFrame::select(
    const std::vector<std::string>& columns) const {
  std::vector<std::shared_ptr<arrow::Field>> fields;
  std::vector<std::shared_ptr<arrow::ChunkedArray>> arrays;

  for (const auto& name : columns) {
    int idx = table->schema()->GetFieldIndex(name);

    if (idx == -1) {
      throw std::runtime_error("Column not found: " + name);
    }

    fields.push_back(table->field(idx));
    arrays.push_back(table->column(idx));  // zero-copy
  }

  auto schema = std::make_shared<arrow::Schema>(fields);
  auto new_table = arrow::Table::Make(schema, arrays);

  return EagerDataFrame(new_table);
}

EagerDataFrame EagerDataFrame::head(int n) const {
  int rows = std::min(n, (int)table->num_rows());
  auto new_table = table->Slice(0, rows);  // zero-copy slice
  return EagerDataFrame(new_table);
}

EagerDataFrame EagerDataFrame::filter(std::shared_ptr<Expr> predicate) const {
  auto mask = predicate->evaluate(table);

  int rows = table->num_rows();
  int cols = table->num_columns();

  std::vector<std::shared_ptr<arrow::Array>> new_arrays;

  // for each column
  for (int j = 0; j < cols; j++) {
    auto column = table->column(j);
    auto chunk = column->chunk(0);

    arrow::Int64Builder int_builder;
    arrow::StringBuilder str_builder;

    bool is_int = (chunk->type()->id() == arrow::Type::INT64);

    for (int i = 0; i < rows; i++) {
      auto keep = std::dynamic_pointer_cast<arrow::BooleanScalar>(mask[i]);

      if (keep->value) {
        auto scalar = chunk->GetScalar(i).ValueOrDie();

        if (is_int) {
          auto val = std::dynamic_pointer_cast<arrow::Int64Scalar>(scalar);
          int_builder.Append(val->value);
        } else {
          auto val = std::dynamic_pointer_cast<arrow::StringScalar>(scalar);
          str_builder.Append(val->ToString());
        }
      }
    }

    std::shared_ptr<arrow::Array> array;

    if (is_int) {
      int_builder.Finish(&array);
    } else {
      str_builder.Finish(&array);
    }

    new_arrays.push_back(array);
  }

  auto new_table = arrow::Table::Make(table->schema(), new_arrays);
  return EagerDataFrame(new_table);
}

EagerDataFrame EagerDataFrame::with_column(const std::string& name,
                                           std::shared_ptr<Expr> expr) const {
  auto values = expr->evaluate(table);

  arrow::Int64Builder builder;

  for (auto& v : values) {
    auto val = std::dynamic_pointer_cast<arrow::Int64Scalar>(v);
    builder.Append(val->value);
  }

  std::shared_ptr<arrow::Array> new_array;
  builder.Finish(&new_array);

  auto chunked_array = std::make_shared<arrow::ChunkedArray>(new_array);

  int idx = table->schema()->GetFieldIndex(name);

  std::shared_ptr<arrow::Table> new_table;

  if (idx == -1) {
    new_table =
        table
            ->AddColumn(table->num_columns(),
                        std::make_shared<arrow::Field>(name, arrow::int64()),
                        chunked_array)
            .ValueOrDie();
  } else {
    new_table =
        table
            ->SetColumn(idx,
                        std::make_shared<arrow::Field>(name, arrow::int64()),
                        chunked_array)
            .ValueOrDie();
  }

  return EagerDataFrame(new_table);
}

EagerDataFrame EagerDataFrame::sort(const std::string& column_name) const {
  int col_idx = table->schema()->GetFieldIndex(column_name);
  if (col_idx == -1) {
    throw std::runtime_error("Column not found: " + column_name);
  }

  auto column = table->column(col_idx);
  auto chunk = column->chunk(0);

  int rows = chunk->length();

  // Step 1: create indices
  std::vector<int> indices(rows);
  for (int i = 0; i < rows; i++) {
    indices[i] = i;
  }

  // Step 2: sort indices based on column values
  std::sort(indices.begin(), indices.end(), [&](int a, int b) {
    auto va = std::dynamic_pointer_cast<arrow::Int64Scalar>(
        chunk->GetScalar(a).ValueOrDie());
    auto vb = std::dynamic_pointer_cast<arrow::Int64Scalar>(
        chunk->GetScalar(b).ValueOrDie());
    return va->value < vb->value;
  });

  // Step 3: rebuild all columns using sorted indices
  std::vector<std::shared_ptr<arrow::ChunkedArray>> new_columns;

  for (int j = 0; j < table->num_columns(); j++) {
    auto col = table->column(j);
    auto ch = col->chunk(0);

    arrow::Int64Builder int_builder;
    arrow::StringBuilder str_builder;

    bool is_int = (ch->type()->id() == arrow::Type::INT64);

    for (int idx : indices) {
      auto scalar = ch->GetScalar(idx).ValueOrDie();

      if (is_int) {
        auto val = std::dynamic_pointer_cast<arrow::Int64Scalar>(scalar);
        int_builder.Append(val->value);
      } else {
        auto val = std::dynamic_pointer_cast<arrow::StringScalar>(scalar);
        str_builder.Append(val->ToString());
      }
    }

    std::shared_ptr<arrow::Array> arr;

    if (is_int) {
      int_builder.Finish(&arr);
    } else {
      str_builder.Finish(&arr);
    }

    new_columns.push_back(std::make_shared<arrow::ChunkedArray>(arr));
  }

  auto new_table = arrow::Table::Make(table->schema(), new_columns);

  return EagerDataFrame(new_table);
}

GroupedDataFrame EagerDataFrame::group_by(
    const std::string& column_name) const {
  int col_idx = table->schema()->GetFieldIndex(column_name);
  if (col_idx == -1) {
    throw std::runtime_error("Column not found");
  }

  auto column = table->column(col_idx);
  auto chunk = column->chunk(0);

  std::map<int64_t, std::vector<int>> groups;

  for (int i = 0; i < chunk->length(); i++) {
    auto scalar = std::dynamic_pointer_cast<arrow::Int64Scalar>(
        chunk->GetScalar(i).ValueOrDie());

    int64_t key = scalar->value;
    groups[key].push_back(i);
  }

  return GroupedDataFrame(table, groups);
}

void GroupedDataFrame::printGroups() const {
  for (const auto& [key, indices] : groups) {
    std::cout << key << " : ";
    for (int idx : indices) {
      std::cout << idx << " ";
    }
    std::cout << "\n";
  }
}

EagerDataFrame GroupedDataFrame::aggregate(const std::string& column_name,
                                           const std::string& op) const {
  int col_idx = table->schema()->GetFieldIndex(column_name);
  if (col_idx == -1) {
    throw std::runtime_error("Column not found");
  }

  auto column = table->column(col_idx);
  auto chunk = column->chunk(0);

  arrow::Int64Builder key_builder;
  arrow::Int64Builder value_builder;

  for (const auto& pair : groups) {
    int64_t key = pair.first;
    const auto& indices = pair.second;

    int64_t result = 0;

    if (op == "sum") {
      for (int idx : indices) {
        auto scalar = std::dynamic_pointer_cast<arrow::Int64Scalar>(
            chunk->GetScalar(idx).ValueOrDie());
        result += scalar->value;
      }
    }

    else if (op == "count") {
      result = indices.size();
    }

    else if (op == "mean") {
      int64_t sum = 0;
      for (int idx : indices) {
        auto scalar = std::dynamic_pointer_cast<arrow::Int64Scalar>(
            chunk->GetScalar(idx).ValueOrDie());
        sum += scalar->value;
      }
      result = sum / indices.size();
    }

    else {
      throw std::runtime_error("Unsupported aggregation");
    }

    key_builder.Append(key);
    value_builder.Append(result);
  }

  std::shared_ptr<arrow::Array> key_array;
  std::shared_ptr<arrow::Array> value_array;

  key_builder.Finish(&key_array);
  value_builder.Finish(&value_array);

  auto key_chunked = std::make_shared<arrow::ChunkedArray>(key_array);
  auto value_chunked = std::make_shared<arrow::ChunkedArray>(value_array);

  auto schema =
      arrow::schema({arrow::field("group_key", arrow::int64()),
                     arrow::field(column_name + "_" + op, arrow::int64())});

  auto new_table = arrow::Table::Make(schema, {key_chunked, value_chunked});

  return EagerDataFrame(new_table);
}

EagerDataFrame EagerDataFrame::join(const EagerDataFrame& other,
                                    const std::string& column_name) const {
  auto left_table = table;
  auto right_table = other.getTable();

  int left_idx = left_table->schema()->GetFieldIndex(column_name);
  int right_idx = right_table->schema()->GetFieldIndex(column_name);

  if (left_idx == -1 || right_idx == -1) {
    throw std::runtime_error("Join column not found");
  }

  auto left_col = left_table->column(left_idx)->chunk(0);
  auto right_col = right_table->column(right_idx)->chunk(0);

  // Step 1: build hashmap for right table
  std::unordered_map<int64_t, int> hash_map;

  for (int i = 0; i < right_col->length(); i++) {
    auto scalar = std::dynamic_pointer_cast<arrow::Int64Scalar>(
        right_col->GetScalar(i).ValueOrDie());
    hash_map[scalar->value] = i;
  }

  // Builders for result
  std::vector<arrow::Int64Builder> int_builders(left_table->num_columns() +
                                                right_table->num_columns() - 1);
  std::vector<arrow::StringBuilder> str_builders(
      left_table->num_columns() + right_table->num_columns() - 1);

  std::vector<bool> is_int;

  // Determine column types
  for (int j = 0; j < left_table->num_columns(); j++) {
    is_int.push_back(left_table->column(j)->type()->id() == arrow::Type::INT64);
  }

  for (int j = 0; j < right_table->num_columns(); j++) {
    if (j == right_idx) continue;  // skip duplicate join column
    is_int.push_back(right_table->column(j)->type()->id() ==
                     arrow::Type::INT64);
  }

  // Step 2: iterate left table
  for (int i = 0; i < left_col->length(); i++) {
    auto left_scalar = std::dynamic_pointer_cast<arrow::Int64Scalar>(
        left_col->GetScalar(i).ValueOrDie());

    int64_t key = left_scalar->value;

    if (hash_map.find(key) == hash_map.end()) continue;

    int right_i = hash_map[key];

    int col_pos = 0;

    // left columns
    for (int j = 0; j < left_table->num_columns(); j++) {
      auto scalar = left_table->column(j)->chunk(0)->GetScalar(i).ValueOrDie();

      if (is_int[col_pos]) {
        auto val = std::dynamic_pointer_cast<arrow::Int64Scalar>(scalar);
        int_builders[col_pos].Append(val->value);
      } else {
        auto val = std::dynamic_pointer_cast<arrow::StringScalar>(scalar);
        str_builders[col_pos].Append(val->ToString());
      }
      col_pos++;
    }

    // right columns (skip join key)
    for (int j = 0; j < right_table->num_columns(); j++) {
      if (j == right_idx) continue;

      auto scalar =
          right_table->column(j)->chunk(0)->GetScalar(right_i).ValueOrDie();

      if (is_int[col_pos]) {
        auto val = std::dynamic_pointer_cast<arrow::Int64Scalar>(scalar);
        int_builders[col_pos].Append(val->value);
      } else {
        auto val = std::dynamic_pointer_cast<arrow::StringScalar>(scalar);
        str_builders[col_pos].Append(val->ToString());
      }
      col_pos++;
    }
  }

  // Step 3: build arrays
  std::vector<std::shared_ptr<arrow::ChunkedArray>> columns;

  for (size_t i = 0; i < is_int.size(); i++) {
    std::shared_ptr<arrow::Array> arr;

    if (is_int[i]) {
      int_builders[i].Finish(&arr);
    } else {
      str_builders[i].Finish(&arr);
    }

    columns.push_back(std::make_shared<arrow::ChunkedArray>(arr));
  }

  // schema
  std::vector<std::shared_ptr<arrow::Field>> fields;

  for (int j = 0; j < left_table->num_columns(); j++) {
    fields.push_back(left_table->field(j));
  }

  for (int j = 0; j < right_table->num_columns(); j++) {
    if (j == right_idx) continue;
    fields.push_back(right_table->field(j));
  }

  auto schema = std::make_shared<arrow::Schema>(fields);

  auto new_table = arrow::Table::Make(schema, columns);

  return EagerDataFrame(new_table);
}