#include "dataframelib/dataframe.h"

#include <arrow/api.h>
#include <arrow/csv/writer.h>
#include <arrow/io/api.h>
#include <parquet/arrow/writer.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace dataframelib {

#define CHECK_ARROW(x)                                     \
  do {                                                     \
    arrow::Status _s = (x);                                \
    if (!_s.ok()) throw std::runtime_error(_s.ToString()); \
  } while (0)

// ========================= HELPERS =========================

static double scalar_to_double(const std::shared_ptr<arrow::Scalar>& s) {
  if (!s || !s->is_valid) return 0.0;
  switch (s->type->id()) {
    case arrow::Type::INT8:
      return (double)std::static_pointer_cast<arrow::Int8Scalar>(s)->value;
    case arrow::Type::INT16:
      return (double)std::static_pointer_cast<arrow::Int16Scalar>(s)->value;
    case arrow::Type::INT32:
      return (double)std::static_pointer_cast<arrow::Int32Scalar>(s)->value;
    case arrow::Type::INT64:
      return (double)std::static_pointer_cast<arrow::Int64Scalar>(s)->value;
    case arrow::Type::UINT8:
      return (double)std::static_pointer_cast<arrow::UInt8Scalar>(s)->value;
    case arrow::Type::UINT16:
      return (double)std::static_pointer_cast<arrow::UInt16Scalar>(s)->value;
    case arrow::Type::UINT32:
      return (double)std::static_pointer_cast<arrow::UInt32Scalar>(s)->value;
    case arrow::Type::UINT64:
      return (double)std::static_pointer_cast<arrow::UInt64Scalar>(s)->value;
    case arrow::Type::FLOAT:
      return (double)std::static_pointer_cast<arrow::FloatScalar>(s)->value;
    case arrow::Type::DOUBLE:
      return std::static_pointer_cast<arrow::DoubleScalar>(s)->value;
    default:
      try {
        return std::stod(s->ToString());
      } catch (...) {
        return 0.0;
      }
  }
}

static bool scalar_is_numeric(const std::shared_ptr<arrow::Scalar>& s) {
  if (!s) return false;
  switch (s->type->id()) {
    case arrow::Type::INT8:
    case arrow::Type::INT16:
    case arrow::Type::INT32:
    case arrow::Type::INT64:
    case arrow::Type::UINT8:
    case arrow::Type::UINT16:
    case arrow::Type::UINT32:
    case arrow::Type::UINT64:
    case arrow::Type::FLOAT:
    case arrow::Type::DOUBLE:
      return true;
    default:
      return false;
  }
}

static bool scalar_is_integer(const std::shared_ptr<arrow::Scalar>& s) {
  if (!s) return false;
  switch (s->type->id()) {
    case arrow::Type::INT8:
    case arrow::Type::INT16:
    case arrow::Type::INT32:
    case arrow::Type::INT64:
    case arrow::Type::UINT8:
    case arrow::Type::UINT16:
    case arrow::Type::UINT32:
    case arrow::Type::UINT64:
      return true;
    default:
      return false;
  }
}

static std::string get_string_value(const std::shared_ptr<arrow::Scalar>& s) {
  if (!s || !s->is_valid) return "";
  if (s->type->id() == arrow::Type::STRING)
    return std::static_pointer_cast<arrow::StringScalar>(s)->value->ToString();
  return s->ToString();
}

// ========================= FAST BUFFER ACCESS =========================
// These bypass GetScalar() heap allocation by reading raw typed buffers directly

// Fast double extraction from any numeric array at index i
static double fast_double(const std::shared_ptr<arrow::Array>& arr, int64_t i) {
  if (arr->IsNull(i)) return 0.0;
  switch (arr->type_id()) {
    case arrow::Type::INT32:  return static_cast<const arrow::Int32Array&>(*arr).Value(i);
    case arrow::Type::INT64:  return (double)static_cast<const arrow::Int64Array&>(*arr).Value(i);
    case arrow::Type::FLOAT:  return static_cast<const arrow::FloatArray&>(*arr).Value(i);
    case arrow::Type::DOUBLE: return static_cast<const arrow::DoubleArray&>(*arr).Value(i);
    case arrow::Type::INT8:   return static_cast<const arrow::Int8Array&>(*arr).Value(i);
    case arrow::Type::INT16:  return static_cast<const arrow::Int16Array&>(*arr).Value(i);
    default: return 0.0;
  }
}

// Fast string extraction
static std::string_view fast_string(const std::shared_ptr<arrow::Array>& arr, int64_t i) {
  if (arr->IsNull(i)) return {};
  if (arr->type_id() == arrow::Type::STRING) {
    auto& sa = static_cast<const arrow::StringArray&>(*arr);
    return sa.GetView(i);
  }
  return {};
}

static bool arr_is_numeric(const std::shared_ptr<arrow::Array>& arr) {
  switch (arr->type_id()) {
    case arrow::Type::INT8: case arrow::Type::INT16: case arrow::Type::INT32:
    case arrow::Type::INT64: case arrow::Type::FLOAT: case arrow::Type::DOUBLE:
      return true;
    default: return false;
  }
}

// Fast comparison without GetScalar
static int fast_compare(const std::shared_ptr<arrow::Array>& arr, int64_t a, int64_t b) {
  bool a_null = arr->IsNull(a), b_null = arr->IsNull(b);
  if (a_null && b_null) return 0;
  if (a_null) return -1;
  if (b_null) return 1;
  if (arr_is_numeric(arr)) {
    double da = fast_double(arr, a), db = fast_double(arr, b);
    return (da < db) ? -1 : (da > db) ? 1 : 0;
  }
  if (arr->type_id() == arrow::Type::STRING) {
    auto& sa = static_cast<const arrow::StringArray&>(*arr);
    auto va = sa.GetView(a), vb = sa.GetView(b);
    return va < vb ? -1 : va > vb ? 1 : 0;
  }
  return 0;
}

// Build group key string without GetScalar
static std::string fast_key(
    const std::vector<std::shared_ptr<arrow::Array>>& key_arrs, int64_t i) {
  std::string key;
  key.reserve(32);
  for (const auto& arr : key_arrs) {
    if (arr->IsNull(i)) { key += "__NULL__|"; continue; }
    if (arr_is_numeric(arr)) {
      // use integer representation for int types to avoid float formatting
      if (arr->type_id() == arrow::Type::INT32 || arr->type_id() == arrow::Type::INT64) {
        key += std::to_string((int64_t)fast_double(arr, i));
      } else {
        key += std::to_string(fast_double(arr, i));
      }
    } else if (arr->type_id() == arrow::Type::STRING) {
      auto& sa = static_cast<const arrow::StringArray&>(*arr);
      auto v = sa.GetView(i);
      key.append(v.data(), v.size());
    }
    key += '|';
  }
  return key;
}

// Build typed Arrow array from scalars
static std::shared_ptr<arrow::Array> scalars_to_array(
    const std::vector<std::shared_ptr<arrow::Scalar>>& scalars) {
  if (scalars.empty()) {
    std::shared_ptr<arrow::Array> out;
    arrow::NullBuilder nb;
    nb.Finish(&out).ok();
    return out;
  }
  std::shared_ptr<arrow::DataType> dtype;
  for (const auto& s : scalars)
    if (s && s->is_valid) {
      dtype = s->type;
      break;
    }
  if (!dtype) dtype = arrow::null();

  switch (dtype->id()) {
    case arrow::Type::BOOL: {
      arrow::BooleanBuilder b;
      b.Reserve(scalars.size()).ok();
      for (const auto& s : scalars) {
        if (!s || !s->is_valid) {
          b.AppendNull().ok();
          continue;
        }
        auto bs = std::dynamic_pointer_cast<arrow::BooleanScalar>(s);
        bs ? b.Append(bs->value).ok() : b.AppendNull().ok();
      }
      std::shared_ptr<arrow::Array> out;
      b.Finish(&out).ok();
      return out;
    }
    case arrow::Type::INT32: {
      arrow::Int32Builder b;
      b.Reserve(scalars.size()).ok();
      for (const auto& s : scalars) {
        if (!s || !s->is_valid)
          b.AppendNull().ok();
        else
          b.Append((int32_t)scalar_to_double(s)).ok();
      }
      std::shared_ptr<arrow::Array> out;
      b.Finish(&out).ok();
      return out;
    }
    case arrow::Type::INT64: {
      arrow::Int64Builder b;
      b.Reserve(scalars.size()).ok();
      for (const auto& s : scalars) {
        if (!s || !s->is_valid) {
          b.AppendNull().ok();
          continue;
        }
        auto is = std::dynamic_pointer_cast<arrow::Int64Scalar>(s);
        is ? b.Append(is->value).ok()
           : b.Append((int64_t)scalar_to_double(s)).ok();
      }
      std::shared_ptr<arrow::Array> out;
      b.Finish(&out).ok();
      return out;
    }
    case arrow::Type::FLOAT: {
      arrow::FloatBuilder b;
      b.Reserve(scalars.size()).ok();
      for (const auto& s : scalars) {
        if (!s || !s->is_valid)
          b.AppendNull().ok();
        else
          b.Append((float)scalar_to_double(s)).ok();
      }
      std::shared_ptr<arrow::Array> out;
      b.Finish(&out).ok();
      return out;
    }
    case arrow::Type::DOUBLE: {
      arrow::DoubleBuilder b;
      b.Reserve(scalars.size()).ok();
      for (const auto& s : scalars) {
        if (!s || !s->is_valid)
          b.AppendNull().ok();
        else
          b.Append(scalar_to_double(s)).ok();
      }
      std::shared_ptr<arrow::Array> out;
      b.Finish(&out).ok();
      return out;
    }
    default: {
      arrow::StringBuilder b;
      b.Reserve(scalars.size()).ok();
      for (const auto& s : scalars) {
        if (!s || !s->is_valid)
          b.AppendNull().ok();
        else
          b.Append(get_string_value(s)).ok();
      }
      std::shared_ptr<arrow::Array> out;
      b.Finish(&out).ok();
      return out;
    }
  }
}

// Extract all scalars from a chunked array
static std::vector<std::shared_ptr<arrow::Scalar>> column_to_scalars(
    const std::shared_ptr<arrow::ChunkedArray>& col) {
  std::vector<std::shared_ptr<arrow::Scalar>> result;
  result.reserve(col->length());
  for (int c = 0; c < col->num_chunks(); c++) {
    auto arr = col->chunk(c);
    for (int64_t i = 0; i < arr->length(); i++)
      result.push_back(arr->GetScalar(i).ValueOrDie());
  }
  return result;
}

// ========================= CTOR =========================
EagerDataFrame::EagerDataFrame(std::shared_ptr<arrow::Table> t)
    : table_(std::move(t)) {}
std::shared_ptr<arrow::Table> EagerDataFrame::getTable() const {
  return table_;
}
int64_t EagerDataFrame::num_rows() const { return table_->num_rows(); }
int EagerDataFrame::num_columns() const { return table_->num_columns(); }

void EagerDataFrame::printSchema() const {
  std::cout << table_->schema()->ToString() << "\n";
}
void EagerDataFrame::printHead(int n) const {
  int rows = (int)std::min((int64_t)n, table_->num_rows());
  for (int i = 0; i < rows; i++) {
    for (int c = 0; c < table_->num_columns(); c++)
      std::cout
          << table_->column(c)->chunk(0)->GetScalar(i).ValueOrDie()->ToString()
          << " ";
    std::cout << "\n";
  }
}

// ========================= SELECT =========================
EagerDataFrame EagerDataFrame::select(
    const std::vector<std::string>& col_names) const {
  std::vector<std::shared_ptr<arrow::Field>> fields;
  std::vector<std::shared_ptr<arrow::ChunkedArray>> arrays;
  fields.reserve(col_names.size());
  arrays.reserve(col_names.size());
  for (const auto& name : col_names) {
    int idx = table_->schema()->GetFieldIndex(name);
    if (idx < 0) throw std::runtime_error("Column not found: " + name);
    fields.push_back(table_->schema()->field(idx));
    arrays.push_back(table_->column(idx));
  }
  return EagerDataFrame(
      arrow::Table::Make(std::make_shared<arrow::Schema>(fields), arrays));
}

// ========================= FILTER =========================
EagerDataFrame EagerDataFrame::filter(ExprPtr predicate) const {
  auto mask = predicate.ptr->evaluate(table_);
  int64_t nrows = table_->num_rows();

  // Collect kept row indices
  std::vector<int64_t> keep_rows;
  keep_rows.reserve(nrows);
  for (int64_t i = 0; i < (int64_t)mask.size(); i++) {
    auto b = std::dynamic_pointer_cast<arrow::BooleanScalar>(mask[i]);
    if (b && b->is_valid && b->value) keep_rows.push_back(i);
  }

  int ncols = table_->num_columns();
  std::vector<std::shared_ptr<arrow::Field>> fields;
  std::vector<std::shared_ptr<arrow::ChunkedArray>> new_cols;
  fields.reserve(ncols);
  new_cols.reserve(ncols);

  for (int c = 0; c < ncols; c++) {
    auto col_scalars = column_to_scalars(table_->column(c));
    std::vector<std::shared_ptr<arrow::Scalar>> filtered;
    filtered.reserve(keep_rows.size());
    for (int64_t row : keep_rows) filtered.push_back(col_scalars[row]);

    auto arr = scalars_to_array(filtered);
    if (arr->type()->id() == arrow::Type::NA && filtered.empty()) {
      auto orig_type = table_->column(c)->type();
      std::shared_ptr<arrow::Array> typed_arr;
      if (orig_type->id() == arrow::Type::INT64) {
        arrow::Int64Builder b;
        b.Finish(&typed_arr).ok();
      } else if (orig_type->id() == arrow::Type::DOUBLE) {
        arrow::DoubleBuilder b;
        b.Finish(&typed_arr).ok();
      } else {
        arrow::StringBuilder b;
        b.Finish(&typed_arr).ok();
      }
      arr = typed_arr;
    }
    fields.push_back(table_->schema()->field(c));
    new_cols.push_back(std::make_shared<arrow::ChunkedArray>(arr));
  }
  return EagerDataFrame(
      arrow::Table::Make(std::make_shared<arrow::Schema>(fields), new_cols));
}

// ========================= WITH COLUMN =========================
EagerDataFrame EagerDataFrame::with_column(const std::string& name,
                                           ExprPtr expr) const {
  auto scalars = expr.ptr->evaluate(table_);
  auto arr = scalars_to_array(scalars);
  auto chunked = std::make_shared<arrow::ChunkedArray>(arr);
  auto field = arrow::field(name, arr->type());
  int idx = table_->schema()->GetFieldIndex(name);
  if (idx >= 0)
    return EagerDataFrame(table_->SetColumn(idx, field, chunked).ValueOrDie());
  return EagerDataFrame(
      table_->AddColumn(table_->num_columns(), field, chunked).ValueOrDie());
}

// ========================= SORT =========================
EagerDataFrame EagerDataFrame::sort(
    const std::vector<std::string>& column_names, bool ascending) const {
  int64_t nrows = table_->num_rows();
  std::vector<int64_t> indices(nrows);
  for (int64_t i = 0; i < nrows; i++) indices[i] = i;

  // Pre-fetch sort columns
  std::vector<std::vector<std::shared_ptr<arrow::Scalar>>> sort_cols;
  sort_cols.reserve(column_names.size());
  for (const auto& cn : column_names) {
    int idx = table_->schema()->GetFieldIndex(cn);
    if (idx < 0) throw std::runtime_error("Sort column not found: " + cn);
    sort_cols.push_back(column_to_scalars(table_->column(idx)));
  }

  std::stable_sort(indices.begin(), indices.end(), [&](int64_t a, int64_t b) {
    for (const auto& sc : sort_cols) {
      const auto& av = sc[a];
      const auto& bv = sc[b];
      bool a_null = !av || !av->is_valid, b_null = !bv || !bv->is_valid;
      if (a_null && b_null) continue;
      if (a_null) return !ascending;
      if (b_null) return ascending;
      int cmp;
      if (scalar_is_numeric(av) && scalar_is_numeric(bv)) {
        double da = scalar_to_double(av), db = scalar_to_double(bv);
        cmp = (da < db) ? -1 : (da > db) ? 1 : 0;
      } else
        cmp = get_string_value(av).compare(get_string_value(bv));
      if (cmp != 0) return ascending ? cmp < 0 : cmp > 0;
    }
    return false;
  });

  int ncols = table_->num_columns();
  std::vector<std::shared_ptr<arrow::Field>> fields;
  std::vector<std::shared_ptr<arrow::ChunkedArray>> new_cols;
  fields.reserve(ncols);
  new_cols.reserve(ncols);
  for (int c = 0; c < ncols; c++) {
    auto col_scalars = column_to_scalars(table_->column(c));
    std::vector<std::shared_ptr<arrow::Scalar>> sorted_scalars;
    sorted_scalars.reserve(nrows);
    for (int64_t i : indices) sorted_scalars.push_back(col_scalars[i]);
    fields.push_back(table_->schema()->field(c));
    new_cols.push_back(std::make_shared<arrow::ChunkedArray>(
        scalars_to_array(sorted_scalars)));
  }
  return EagerDataFrame(
      arrow::Table::Make(std::make_shared<arrow::Schema>(fields), new_cols));
}

// ========================= SORT + LIMIT FUSION =========================
EagerDataFrame EagerDataFrame::sort_head(
    const std::vector<std::string>& column_names, bool ascending,
    int64_t n) const {
  int64_t nrows = table_->num_rows();
  n = std::min(n, nrows);
  std::vector<int64_t> indices(nrows);
  for (int64_t i = 0; i < nrows; i++) indices[i] = i;

  std::vector<std::vector<std::shared_ptr<arrow::Scalar>>> sort_cols;
  sort_cols.reserve(column_names.size());
  for (const auto& cn : column_names) {
    int idx = table_->schema()->GetFieldIndex(cn);
    if (idx < 0) throw std::runtime_error("Sort column not found: " + cn);
    sort_cols.push_back(column_to_scalars(table_->column(idx)));
  }

  auto cmp = [&](int64_t a, int64_t b) {
    for (const auto& sc : sort_cols) {
      const auto& av = sc[a];
      const auto& bv = sc[b];
      bool a_null = !av || !av->is_valid, b_null = !bv || !bv->is_valid;
      if (a_null && b_null) continue;
      if (a_null) return !ascending;
      if (b_null) return ascending;
      int cv;
      if (scalar_is_numeric(av) && scalar_is_numeric(bv)) {
        double da = scalar_to_double(av), db = scalar_to_double(bv);
        cv = (da < db) ? -1 : (da > db) ? 1 : 0;
      } else
        cv = get_string_value(av).compare(get_string_value(bv));
      if (cv != 0) return ascending ? cv < 0 : cv > 0;
    }
    return false;
  };

  // partial_sort: O(N log n) vs O(N log N) for full sort
  std::partial_sort(indices.begin(), indices.begin() + n, indices.end(), cmp);
  indices.resize(n);

  int ncols = table_->num_columns();
  std::vector<std::shared_ptr<arrow::Field>> fields;
  std::vector<std::shared_ptr<arrow::ChunkedArray>> new_cols;
  fields.reserve(ncols);
  new_cols.reserve(ncols);
  for (int c = 0; c < ncols; c++) {
    auto col_scalars = column_to_scalars(table_->column(c));
    std::vector<std::shared_ptr<arrow::Scalar>> out;
    out.reserve(n);
    for (int64_t i : indices) out.push_back(col_scalars[i]);
    fields.push_back(table_->schema()->field(c));
    new_cols.push_back(
        std::make_shared<arrow::ChunkedArray>(scalars_to_array(out)));
  }
  return EagerDataFrame(
      arrow::Table::Make(std::make_shared<arrow::Schema>(fields), new_cols));
}

// ========================= HEAD =========================
EagerDataFrame EagerDataFrame::head(int n) const {
  return EagerDataFrame(
      table_->Slice(0, std::min((int64_t)n, table_->num_rows())));
}

// ========================= GROUP BY =========================
// Uses unordered_map for O(1) average lookup instead of O(log n) with map
GroupedDataFrame EagerDataFrame::group_by(
    const std::vector<std::string>& column_names) const {
  std::vector<std::vector<std::shared_ptr<arrow::Scalar>>> key_cols;
  key_cols.reserve(column_names.size());
  for (const auto& cn : column_names) {
    int idx = table_->schema()->GetFieldIndex(cn);
    if (idx < 0) throw std::runtime_error("Group-by column not found: " + cn);
    key_cols.push_back(column_to_scalars(table_->column(idx)));
  }

  std::unordered_map<std::string, std::vector<int>> groups;
  groups.reserve(1024);
  std::vector<std::string> key_order;  // preserves insertion order

  for (int64_t i = 0; i < table_->num_rows(); i++) {
    std::string key;
    key.reserve(64);
    for (const auto& kc : key_cols) {
      const auto& s = kc[i];
      if (!s || !s->is_valid)
        key += "__NULL__|";
      else {
        key += get_string_value(s);
        key += '|';
      }
    }
    auto it = groups.find(key);
    if (it == groups.end()) {
      key_order.push_back(key);
      groups[key] = {(int)i};
    } else {
      it->second.push_back((int)i);
    }
  }

  // Convert unordered_map to map (required by GroupedDataFrame)
  std::map<std::string, std::vector<int>> ordered(groups.begin(), groups.end());
  return GroupedDataFrame(table_, column_names, std::move(ordered));
}

// ========================= AGGREGATE =========================
EagerDataFrame GroupedDataFrame::aggregate(
    const std::vector<std::pair<std::string, std::string>>& agg_map) const {
  struct AggSpec {
    std::string output_name, func, input_col;
  };
  std::vector<AggSpec> specs;
  specs.reserve(agg_map.size());

  for (const auto& [out, spec] : agg_map) {
    AggSpec s;
    bool spec_is_func = (spec == "sum" || spec == "mean" || spec == "count" ||
                         spec == "min" || spec == "max");
    if (spec_is_func) {
      s.func = spec;
      s.input_col = out;
      std::string expected_suffix = "_" + spec;
      if (out.size() > expected_suffix.size() &&
          out.substr(out.size() - expected_suffix.size()) == expected_suffix) {
        s.input_col = out.substr(0, out.size() - expected_suffix.size());
        s.output_name = out;
      } else {
        s.output_name = out + "_" + spec;
      }
    } else {
      auto colon = spec.find(':');
      if (colon != std::string::npos) {
        s.func = spec.substr(0, colon);
        s.input_col = spec.substr(colon + 1);
        s.output_name = out;
      } else {
        s.input_col = spec;
        s.output_name = out;
        for (const char* suf : {"_sum", "_mean", "_count", "_min", "_max"}) {
          size_t slen = strlen(suf);
          if (out.size() > slen && out.substr(out.size() - slen) == suf) {
            if (std::string(suf) == "_sum")
              s.func = "sum";
            else if (std::string(suf) == "_mean")
              s.func = "mean";
            else if (std::string(suf) == "_count")
              s.func = "count";
            else if (std::string(suf) == "_min")
              s.func = "min";
            else if (std::string(suf) == "_max")
              s.func = "max";
            break;
          }
        }
        if (s.func.empty()) s.func = "sum";
      }
    }
    specs.push_back(s);
  }

  // Pre-fetch input columns once
  std::unordered_map<std::string, std::vector<std::shared_ptr<arrow::Scalar>>>
      col_cache;
  col_cache.reserve(specs.size());
  for (const auto& sp : specs) {
    if (!col_cache.count(sp.input_col)) {
      int idx = table_->schema()->GetFieldIndex(sp.input_col);
      if (idx >= 0)
        col_cache[sp.input_col] = column_to_scalars(table_->column(idx));
    }
  }

  // Pre-fetch key columns once
  std::unordered_map<std::string, std::vector<std::shared_ptr<arrow::Scalar>>>
      key_col_cache;
  key_col_cache.reserve(keys_.size());
  for (const auto& kn : keys_) {
    int idx = table_->schema()->GetFieldIndex(kn);
    if (idx >= 0) key_col_cache[kn] = column_to_scalars(table_->column(idx));
  }

  // Use pre-computed insertion-ordered keys from group_by
  // key_order is passed from group_by via groups_ iteration order
  std::vector<std::string> key_order;
  {
    std::unordered_map<std::string,bool> seen; seen.reserve(groups_.size());
    for(const auto& [k,v]: groups_) { if(!seen.count(k)){seen[k]=true; key_order.push_back(k);} }
    // Sort by first row index to get insertion order
    std::sort(key_order.begin(), key_order.end(), [&](const std::string& a, const std::string& b){
      return groups_.at(a)[0] < groups_.at(b)[0];
    });
  }

  // Build output schema
  std::vector<std::shared_ptr<arrow::Field>> fields;
  fields.reserve(keys_.size() + specs.size());
  for (const auto& k : keys_) {
    int idx = table_->schema()->GetFieldIndex(k);
    fields.push_back(table_->schema()->field(idx));
  }
  for (const auto& sp : specs) {
    std::shared_ptr<arrow::DataType> out_type;
    if (sp.func == "count")
      out_type = arrow::int64();
    else if (sp.func == "sum") {
      int idx = table_->schema()->GetFieldIndex(sp.input_col);
      if (idx >= 0) {
        auto t = table_->schema()->field(idx)->type();
        out_type =
            (t->id() == arrow::Type::INT32 || t->id() == arrow::Type::INT64)
                ? arrow::int64()
                : arrow::float64();
      } else
        out_type = arrow::float64();
    } else if (sp.func == "mean") {
      out_type = arrow::float64();
    } else {
      int idx = table_->schema()->GetFieldIndex(sp.input_col);
      out_type =
          idx >= 0 ? table_->schema()->field(idx)->type() : arrow::float64();
    }
    fields.push_back(arrow::field(sp.output_name, out_type));
  }

  size_t ngroups = key_order.size();
  std::vector<std::vector<std::shared_ptr<arrow::Scalar>>> col_scalars(
      fields.size());
  for (auto& v : col_scalars) v.reserve(ngroups);

  for (size_t g = 0; g < ngroups; g++) {
    const auto& rows = groups_.at(key_order[g]);

    for (size_t k = 0; k < keys_.size(); k++)
      col_scalars[k].push_back(key_col_cache[keys_[k]][rows[0]]);

    for (size_t a = 0; a < specs.size(); a++) {
      const auto& sp = specs[a];
      const auto& col_data =
          col_cache.count(sp.input_col)
              ? col_cache.at(sp.input_col)
              : std::vector<std::shared_ptr<arrow::Scalar>>{};
      std::shared_ptr<arrow::Scalar> result;

      if (sp.func == "sum") {
        double sum = 0.0;
        bool any = false;
        for (int r : rows)
          if (r < (int)col_data.size() && col_data[r] &&
              col_data[r]->is_valid) {
            sum += scalar_to_double(col_data[r]);
            any = true;
          }
        if (any) {
          auto t = fields[keys_.size() + a]->type();
          result = (t->id() == arrow::Type::INT64)
                       ? arrow::MakeScalar((int64_t)sum)
                       : arrow::MakeScalar(sum);
        } else
          result = arrow::MakeNullScalar(arrow::float64());
      } else if (sp.func == "mean") {
        double sum = 0.0;
        int64_t cnt = 0;
        for (int r : rows)
          if (r < (int)col_data.size() && col_data[r] &&
              col_data[r]->is_valid) {
            sum += scalar_to_double(col_data[r]);
            cnt++;
          }
        result = cnt > 0 ? arrow::MakeScalar(sum / cnt)
                         : arrow::MakeNullScalar(arrow::float64());
      } else if (sp.func == "count") {
        int64_t cnt = 0;
        for (int r : rows)
          if (r < (int)col_data.size() && col_data[r] && col_data[r]->is_valid)
            cnt++;
        result = arrow::MakeScalar(cnt);
      } else if (sp.func == "min") {
        std::shared_ptr<arrow::Scalar> mn;
        for (int r : rows)
          if (r < (int)col_data.size() && col_data[r] && col_data[r]->is_valid)
            if (!mn || scalar_to_double(col_data[r]) < scalar_to_double(mn))
              mn = col_data[r];
        result = mn ? mn : arrow::MakeNullScalar(arrow::float64());
      } else if (sp.func == "max") {
        std::shared_ptr<arrow::Scalar> mx;
        for (int r : rows)
          if (r < (int)col_data.size() && col_data[r] && col_data[r]->is_valid)
            if (!mx || scalar_to_double(col_data[r]) > scalar_to_double(mx))
              mx = col_data[r];
        result = mx ? mx : arrow::MakeNullScalar(arrow::float64());
      } else
        result = arrow::MakeNullScalar(arrow::float64());

      col_scalars[keys_.size() + a].push_back(result);
    }
  }

  // Build typed arrays matching schema exactly
  auto schema = std::make_shared<arrow::Schema>(fields);
  std::vector<std::shared_ptr<arrow::ChunkedArray>> arrays;
  arrays.reserve(fields.size());
  for (size_t c = 0; c < fields.size(); c++) {
    auto target_type = fields[c]->type();
    const auto& scalars = col_scalars[c];
    std::shared_ptr<arrow::Array> arr;
    if (target_type->id() == arrow::Type::INT64) {
      arrow::Int64Builder b;
      b.Reserve(scalars.size()).ok();
      for (const auto& s : scalars) {
        if (!s || !s->is_valid)
          b.AppendNull().ok();
        else {
          auto is = std::dynamic_pointer_cast<arrow::Int64Scalar>(s);
          is ? b.Append(is->value).ok()
             : b.Append((int64_t)scalar_to_double(s)).ok();
        }
      }
      b.Finish(&arr).ok();
    } else if (target_type->id() == arrow::Type::DOUBLE) {
      arrow::DoubleBuilder b;
      b.Reserve(scalars.size()).ok();
      for (const auto& s : scalars) {
        if (!s || !s->is_valid)
          b.AppendNull().ok();
        else
          b.Append(scalar_to_double(s)).ok();
      }
      b.Finish(&arr).ok();
    } else if (target_type->id() == arrow::Type::INT32) {
      arrow::Int32Builder b;
      b.Reserve(scalars.size()).ok();
      for (const auto& s : scalars) {
        if (!s || !s->is_valid)
          b.AppendNull().ok();
        else
          b.Append((int32_t)scalar_to_double(s)).ok();
      }
      b.Finish(&arr).ok();
    } else {
      arrow::StringBuilder b;
      b.Reserve(scalars.size()).ok();
      for (const auto& s : scalars) {
        if (!s || !s->is_valid)
          b.AppendNull().ok();
        else
          b.Append(get_string_value(s)).ok();
      }
      b.Finish(&arr).ok();
    }
    arrays.push_back(std::make_shared<arrow::ChunkedArray>(arr));
  }
  return EagerDataFrame(arrow::Table::Make(schema, arrays));
}

// ========================= JOIN =========================
EagerDataFrame EagerDataFrame::join(
    const EagerDataFrame& other, const std::vector<std::string>& column_names,
    const std::string& how) const {
  if (how != "inner" && how != "left" && how != "outer" && how != "right")
    throw std::runtime_error("Unsupported join: " + how);

  auto left_table = table_, right_table = other.table_;

  auto make_keys = [](const std::shared_ptr<arrow::Table>& t,
                      const std::vector<std::string>& cols) {
    std::vector<std::vector<std::shared_ptr<arrow::Scalar>>> key_cols;
    key_cols.reserve(cols.size());
    for (const auto& cn : cols) {
      int idx = t->schema()->GetFieldIndex(cn);
      if (idx < 0) throw std::runtime_error("Join key not found: " + cn);
      key_cols.push_back(column_to_scalars(t->column(idx)));
    }
    std::vector<std::string> keys(t->num_rows());
    for (int64_t i = 0; i < t->num_rows(); i++) {
      keys[i].reserve(64);
      for (const auto& kc : key_cols) {
        if (!kc[i] || !kc[i]->is_valid)
          keys[i] += "__NULL__|";
        else {
          keys[i] += get_string_value(kc[i]);
          keys[i] += '|';
        }
      }
    }
    return keys;
  };

  auto left_keys = make_keys(left_table, column_names);
  auto right_keys = make_keys(right_table, column_names);

  // Use unordered_map for O(1) join lookups
  std::unordered_map<std::string, std::vector<int64_t>> right_index;
  right_index.reserve(right_keys.size());
  for (int64_t i = 0; i < (int64_t)right_keys.size(); i++)
    right_index[right_keys[i]].push_back(i);

  std::set<std::string> key_set(column_names.begin(), column_names.end());
  std::vector<std::shared_ptr<arrow::Field>> fields;
  for (int c = 0; c < left_table->num_columns(); c++)
    fields.push_back(left_table->schema()->field(c));
  std::vector<int> right_extra;
  for (int c = 0; c < right_table->num_columns(); c++) {
    if (!key_set.count(right_table->schema()->field(c)->name())) {
      right_extra.push_back(c);
      fields.push_back(right_table->schema()->field(c));
    }
  }

  std::vector<std::vector<std::shared_ptr<arrow::Scalar>>> left_data,
      right_data;
  left_data.reserve(left_table->num_columns());
  right_data.reserve(right_extra.size());
  for (int c = 0; c < left_table->num_columns(); c++)
    left_data.push_back(column_to_scalars(left_table->column(c)));
  for (int c : right_extra)
    right_data.push_back(column_to_scalars(right_table->column(c)));

  std::vector<std::vector<std::shared_ptr<arrow::Scalar>>> out_cols(
      fields.size());
  std::set<int64_t> matched_right;

  auto add_row = [&](int64_t l, int64_t r) {
    for (int c = 0; c < left_table->num_columns(); c++)
      out_cols[c].push_back(l >= 0 ? left_data[c][l]
                                   : arrow::MakeNullScalar(fields[c]->type()));
    for (size_t rc = 0; rc < right_extra.size(); rc++)
      out_cols[left_table->num_columns() + rc].push_back(
          r >= 0 ? right_data[rc][r]
                 : arrow::MakeNullScalar(
                       fields[left_table->num_columns() + rc]->type()));
  };

  for (int64_t l = 0; l < (int64_t)left_keys.size(); l++) {
    auto it = right_index.find(left_keys[l]);
    if (it != right_index.end()) {
      for (int64_t r : it->second) {
        add_row(l, r);
        matched_right.insert(r);
      }
    } else if (how == "left" || how == "outer")
      add_row(l, -1);
  }

  if (how == "outer") {
    for (int64_t r = 0; r < (int64_t)right_keys.size(); r++) {
      if (!matched_right.count(r)) {
        add_row(-1, r);
        for (const auto& kn : column_names) {
          int l_idx = left_table->schema()->GetFieldIndex(kn);
          int r_idx = right_table->schema()->GetFieldIndex(kn);
          if (l_idx >= 0 && r_idx >= 0)
            out_cols[l_idx].back() =
                column_to_scalars(right_table->column(r_idx))[r];
        }
      }
    }
  }

  std::vector<std::shared_ptr<arrow::ChunkedArray>> arrays;
  arrays.reserve(fields.size());
  for (size_t c = 0; c < fields.size(); c++)
    arrays.push_back(
        std::make_shared<arrow::ChunkedArray>(scalars_to_array(out_cols[c])));

  return EagerDataFrame(
      arrow::Table::Make(std::make_shared<arrow::Schema>(fields), arrays));
}

// ========================= WRITE =========================
void EagerDataFrame::write_csv(const std::string& path) const {
  auto out = arrow::io::FileOutputStream::Open(path).ValueOrDie();
  CHECK_ARROW(arrow::csv::WriteCSV(
      *table_, arrow::csv::WriteOptions::Defaults(), out.get()));
}
void EagerDataFrame::write_parquet(const std::string& path) const {
  auto out = arrow::io::FileOutputStream::Open(path).ValueOrDie();
  auto writer = parquet::arrow::FileWriter::Open(
                    *table_->schema(), arrow::default_memory_pool(), out)
                    .ValueOrDie();
  CHECK_ARROW(writer->WriteTable(*table_, table_->num_rows()));
  CHECK_ARROW(writer->Close());
}

}  // namespace dataframelib