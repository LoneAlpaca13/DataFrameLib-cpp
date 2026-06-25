#include "dataframelib/expr.h"

#include <arrow/api.h>
#include <arrow/scalar.h>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <cctype>

namespace dataframelib {

// ========================= HELPERS =========================

static double scalar_to_double(const std::shared_ptr<arrow::Scalar>& s) {
  if (!s || !s->is_valid) return 0.0;
  switch (s->type->id()) {
    case arrow::Type::INT8:   return (double)std::static_pointer_cast<arrow::Int8Scalar>(s)->value;
    case arrow::Type::INT16:  return (double)std::static_pointer_cast<arrow::Int16Scalar>(s)->value;
    case arrow::Type::INT32:  return (double)std::static_pointer_cast<arrow::Int32Scalar>(s)->value;
    case arrow::Type::INT64:  return (double)std::static_pointer_cast<arrow::Int64Scalar>(s)->value;
    case arrow::Type::UINT8:  return (double)std::static_pointer_cast<arrow::UInt8Scalar>(s)->value;
    case arrow::Type::UINT16: return (double)std::static_pointer_cast<arrow::UInt16Scalar>(s)->value;
    case arrow::Type::UINT32: return (double)std::static_pointer_cast<arrow::UInt32Scalar>(s)->value;
    case arrow::Type::UINT64: return (double)std::static_pointer_cast<arrow::UInt64Scalar>(s)->value;
    case arrow::Type::FLOAT:  return (double)std::static_pointer_cast<arrow::FloatScalar>(s)->value;
    case arrow::Type::DOUBLE: return std::static_pointer_cast<arrow::DoubleScalar>(s)->value;
    default: {
      try { return std::stod(s->ToString()); } catch (...) { return 0.0; }
    }
  }
}

static int64_t scalar_to_int64(const std::shared_ptr<arrow::Scalar>& s) {
  if (!s || !s->is_valid) return 0;
  switch (s->type->id()) {
    case arrow::Type::INT8:   return std::static_pointer_cast<arrow::Int8Scalar>(s)->value;
    case arrow::Type::INT16:  return std::static_pointer_cast<arrow::Int16Scalar>(s)->value;
    case arrow::Type::INT32:  return std::static_pointer_cast<arrow::Int32Scalar>(s)->value;
    case arrow::Type::INT64:  return std::static_pointer_cast<arrow::Int64Scalar>(s)->value;
    case arrow::Type::UINT8:  return std::static_pointer_cast<arrow::UInt8Scalar>(s)->value;
    case arrow::Type::UINT16: return std::static_pointer_cast<arrow::UInt16Scalar>(s)->value;
    case arrow::Type::UINT32: return std::static_pointer_cast<arrow::UInt32Scalar>(s)->value;
    case arrow::Type::UINT64: return (int64_t)std::static_pointer_cast<arrow::UInt64Scalar>(s)->value;
    case arrow::Type::FLOAT:  return (int64_t)std::static_pointer_cast<arrow::FloatScalar>(s)->value;
    case arrow::Type::DOUBLE: return (int64_t)std::static_pointer_cast<arrow::DoubleScalar>(s)->value;
    default: {
      try { return (int64_t)std::stoll(s->ToString()); } catch (...) { return 0; }
    }
  }
}

static bool scalar_is_numeric(const std::shared_ptr<arrow::Scalar>& s) {
  if (!s) return false;
  switch (s->type->id()) {
    case arrow::Type::INT8: case arrow::Type::INT16: case arrow::Type::INT32:
    case arrow::Type::INT64: case arrow::Type::UINT8: case arrow::Type::UINT16:
    case arrow::Type::UINT32: case arrow::Type::UINT64:
    case arrow::Type::FLOAT: case arrow::Type::DOUBLE:
      return true;
    default: return false;
  }
}

static bool scalar_is_integer(const std::shared_ptr<arrow::Scalar>& s) {
  if (!s) return false;
  switch (s->type->id()) {
    case arrow::Type::INT8: case arrow::Type::INT16: case arrow::Type::INT32:
    case arrow::Type::INT64: case arrow::Type::UINT8: case arrow::Type::UINT16:
    case arrow::Type::UINT32: case arrow::Type::UINT64:
      return true;
    default: return false;
  }
}

static std::shared_ptr<arrow::Scalar> make_null() {
  return arrow::MakeNullScalar(arrow::null());
}

static std::shared_ptr<arrow::Scalar> make_double(double v) {
  return arrow::MakeScalar(v);
}
static std::shared_ptr<arrow::Scalar> make_int64(int64_t v) {
  return arrow::MakeScalar(v);
}
static std::shared_ptr<arrow::Scalar> make_bool(bool v) {
  return arrow::MakeScalar(v);
}

static std::string scalar_to_string(const std::shared_ptr<arrow::Scalar>& s) {
  if (!s || !s->is_valid) return "";
  if (s->type->id() == arrow::Type::STRING) {
    return std::static_pointer_cast<arrow::StringScalar>(s)->ToString();
  }
  // For string scalars Arrow wraps them in quotes in ToString()
  // Need raw value
  if (s->type->id() == arrow::Type::LARGE_STRING) {
    return std::static_pointer_cast<arrow::LargeStringScalar>(s)->ToString();
  }
  return s->ToString();
}

static std::string get_string_value(const std::shared_ptr<arrow::Scalar>& s) {
  if (!s || !s->is_valid) return "";
  if (s->type->id() == arrow::Type::STRING) {
    auto ss = std::static_pointer_cast<arrow::StringScalar>(s);
    return ss->value->ToString();
  }
  return s->ToString();
}

// Arithmetic scalar op
static std::shared_ptr<arrow::Scalar> arith_op(
    const std::shared_ptr<arrow::Scalar>& lv,
    const std::shared_ptr<arrow::Scalar>& rv,
    OpType op) {
  if (!lv || !lv->is_valid || !rv || !rv->is_valid) {
    return arrow::MakeNullScalar(arrow::float64());
  }

  bool l_int = scalar_is_integer(lv);
  bool r_int = scalar_is_integer(rv);

  if (l_int && r_int && op == OpType::MOD) {
    int64_t a = scalar_to_int64(lv);
    int64_t b = scalar_to_int64(rv);
    if (b == 0) return arrow::MakeNullScalar(arrow::int64());
    return make_int64(a % b);
  }

  double a = scalar_to_double(lv);
  double b = scalar_to_double(rv);

  switch (op) {
    case OpType::ADD: {
      double res = a + b;
      if (l_int && r_int) return make_int64((int64_t)res);
      return make_double(res);
    }
    case OpType::SUB: {
      double res = a - b;
      if (l_int && r_int) return make_int64((int64_t)res);
      return make_double(res);
    }
    case OpType::MUL: {
      double res = a * b;
      if (l_int && r_int) return make_int64((int64_t)res);
      return make_double(res);
    }
    case OpType::DIV: {
      if (b == 0.0) return arrow::MakeNullScalar(arrow::float64());
      return make_double(a / b);
    }
    case OpType::MOD: {
      if (b == 0.0) return arrow::MakeNullScalar(arrow::float64());
      return make_double(std::fmod(a, b));
    }
    default: return arrow::MakeNullScalar(arrow::float64());
  }
}

static int compare_scalars(const std::shared_ptr<arrow::Scalar>& lv,
                            const std::shared_ptr<arrow::Scalar>& rv) {
  // -1 = l < r, 0 = equal, 1 = l > r
  if (!lv || !lv->is_valid) return -1;
  if (!rv || !rv->is_valid) return 1;

  bool l_num = scalar_is_numeric(lv);
  bool r_num = scalar_is_numeric(rv);

  if (l_num && r_num) {
    double a = scalar_to_double(lv);
    double b = scalar_to_double(rv);
    if (a < b) return -1;
    if (a > b) return 1;
    return 0;
  }

  // String comparison
  std::string a = get_string_value(lv);
  std::string b = get_string_value(rv);
  if (a < b) return -1;
  if (a > b) return 1;
  return 0;
}

// ========================= LITERAL EVALUATE =========================
std::vector<std::shared_ptr<arrow::Scalar>> LiteralExpr::evaluate(
    const std::shared_ptr<arrow::Table>& table) const {
  int64_t n = table ? table->num_rows() : 1;
  return std::vector<std::shared_ptr<arrow::Scalar>>(n, value);
}

// ========================= COLUMN EVALUATE =========================
std::vector<std::shared_ptr<arrow::Scalar>> ColumnExpr::evaluate(
    const std::shared_ptr<arrow::Table>& table) const {
  int idx = table->schema()->GetFieldIndex(name);
  if (idx < 0) throw std::runtime_error("Column not found: " + name);

  auto chunked = table->column(idx);
  std::vector<std::shared_ptr<arrow::Scalar>> result;
  result.reserve(table->num_rows());

  for (int c = 0; c < chunked->num_chunks(); c++) {
    auto arr = chunked->chunk(c);
    for (int64_t i = 0; i < arr->length(); i++) {
      result.push_back(arr->GetScalar(i).ValueOrDie());
    }
  }
  return result;
}

// ========================= UNARY EVALUATE =========================
std::vector<std::shared_ptr<arrow::Scalar>> UnaryExpr::evaluate(
    const std::shared_ptr<arrow::Table>& table) const {
  auto vals = operand->evaluate(table);
  std::vector<std::shared_ptr<arrow::Scalar>> result;
  result.reserve(vals.size());

  for (const auto& v : vals) {
    switch (op) {
      case OpType::NOT: {
        if (!v || !v->is_valid) {
          result.push_back(arrow::MakeNullScalar(arrow::boolean()));
        } else {
          auto b = std::dynamic_pointer_cast<arrow::BooleanScalar>(v);
          if (b) result.push_back(make_bool(!b->value));
          else result.push_back(make_bool(false));
        }
        break;
      }
      case OpType::ABS: {
        if (!v || !v->is_valid) {
          result.push_back(arrow::MakeNullScalar(arrow::float64()));
        } else {
          double val = scalar_to_double(v);
          if (scalar_is_integer(v))
            result.push_back(make_int64(std::abs(scalar_to_int64(v))));
          else
            result.push_back(make_double(std::abs(val)));
        }
        break;
      }
      case OpType::IS_NULL: {
        result.push_back(make_bool(!v || !v->is_valid));
        break;
      }
      case OpType::IS_NOT_NULL: {
        result.push_back(make_bool(v && v->is_valid));
        break;
      }
      case OpType::STR_UPPER: {
        if (!v || !v->is_valid) {
          result.push_back(arrow::MakeNullScalar(arrow::utf8()));
        } else {
          std::string s = get_string_value(v);
          std::transform(s.begin(), s.end(), s.begin(), ::toupper);
          result.push_back(arrow::MakeScalar(s));
        }
        break;
      }
      case OpType::STR_LOWER: {
        if (!v || !v->is_valid) {
          result.push_back(arrow::MakeNullScalar(arrow::utf8()));
        } else {
          std::string s = get_string_value(v);
          std::transform(s.begin(), s.end(), s.begin(), ::tolower);
          result.push_back(arrow::MakeScalar(s));
        }
        break;
      }
      case OpType::STR_LENGTH: {
        if (!v || !v->is_valid) {
          result.push_back(arrow::MakeNullScalar(arrow::int64()));
        } else {
          std::string s = get_string_value(v);
          result.push_back(make_int64((int64_t)s.size()));
        }
        break;
      }
      case OpType::STR_CONTAINS: {
        if (!v || !v->is_valid) {
          result.push_back(arrow::MakeNullScalar(arrow::boolean()));
        } else {
          std::string s = get_string_value(v);
          result.push_back(make_bool(s.find(str_arg) != std::string::npos));
        }
        break;
      }
      case OpType::STR_STARTSWITH: {
        if (!v || !v->is_valid) {
          result.push_back(arrow::MakeNullScalar(arrow::boolean()));
        } else {
          std::string s = get_string_value(v);
          result.push_back(make_bool(s.substr(0, str_arg.size()) == str_arg));
        }
        break;
      }
      case OpType::STR_ENDSWITH: {
        if (!v || !v->is_valid) {
          result.push_back(arrow::MakeNullScalar(arrow::boolean()));
        } else {
          std::string s = get_string_value(v);
          if (s.size() >= str_arg.size())
            result.push_back(make_bool(s.substr(s.size() - str_arg.size()) == str_arg));
          else
            result.push_back(make_bool(false));
        }
        break;
      }
      default:
        result.push_back(v);
    }
  }
  return result;
}

// ========================= BINARY EVALUATE =========================
std::vector<std::shared_ptr<arrow::Scalar>> BinaryExpr::evaluate(
    const std::shared_ptr<arrow::Table>& table) const {
  auto lvals = left_->evaluate(table);
  auto rvals = right_->evaluate(table);

  size_t n = lvals.size();
  std::vector<std::shared_ptr<arrow::Scalar>> result;
  result.reserve(n);

  for (size_t i = 0; i < n; i++) {
    const auto& lv = lvals[i];
    const auto& rv = rvals[i];

    switch (op_) {
      case OpType::ADD:
      case OpType::SUB:
      case OpType::MUL:
      case OpType::DIV:
      case OpType::MOD:
        result.push_back(arith_op(lv, rv, op_));
        break;

      case OpType::EQ: {
        if (!lv || !lv->is_valid || !rv || !rv->is_valid)
          result.push_back(arrow::MakeNullScalar(arrow::boolean()));
        else
          result.push_back(make_bool(compare_scalars(lv, rv) == 0));
        break;
      }
      case OpType::NEQ: {
        if (!lv || !lv->is_valid || !rv || !rv->is_valid)
          result.push_back(arrow::MakeNullScalar(arrow::boolean()));
        else
          result.push_back(make_bool(compare_scalars(lv, rv) != 0));
        break;
      }
      case OpType::GT: {
        if (!lv || !lv->is_valid || !rv || !rv->is_valid)
          result.push_back(arrow::MakeNullScalar(arrow::boolean()));
        else
          result.push_back(make_bool(compare_scalars(lv, rv) > 0));
        break;
      }
      case OpType::LT: {
        if (!lv || !lv->is_valid || !rv || !rv->is_valid)
          result.push_back(arrow::MakeNullScalar(arrow::boolean()));
        else
          result.push_back(make_bool(compare_scalars(lv, rv) < 0));
        break;
      }
      case OpType::GTE: {
        if (!lv || !lv->is_valid || !rv || !rv->is_valid)
          result.push_back(arrow::MakeNullScalar(arrow::boolean()));
        else
          result.push_back(make_bool(compare_scalars(lv, rv) >= 0));
        break;
      }
      case OpType::LTE: {
        if (!lv || !lv->is_valid || !rv || !rv->is_valid)
          result.push_back(arrow::MakeNullScalar(arrow::boolean()));
        else
          result.push_back(make_bool(compare_scalars(lv, rv) <= 0));
        break;
      }
      case OpType::AND: {
        auto lb = std::dynamic_pointer_cast<arrow::BooleanScalar>(lv);
        auto rb = std::dynamic_pointer_cast<arrow::BooleanScalar>(rv);
        if (!lb || !rb) result.push_back(make_bool(false));
        else result.push_back(make_bool(lb->value && rb->value));
        break;
      }
      case OpType::OR: {
        auto lb = std::dynamic_pointer_cast<arrow::BooleanScalar>(lv);
        auto rb = std::dynamic_pointer_cast<arrow::BooleanScalar>(rv);
        if (!lb && !rb) result.push_back(make_bool(false));
        else if (!lb) result.push_back(make_bool(rb && rb->value));
        else if (!rb) result.push_back(make_bool(lb && lb->value));
        else result.push_back(make_bool(lb->value || rb->value));
        break;
      }
      default:
        result.push_back(lv);
    }
  }
  return result;
}

// ========================= AGG EVALUATE =========================
// For aggregation, we return a single scalar repeated (used in group_by context)
std::vector<std::shared_ptr<arrow::Scalar>> AggExpr::evaluate(
    const std::shared_ptr<arrow::Table>& table) const {
  auto vals = input->evaluate(table);
  if (vals.empty()) {
    return {arrow::MakeNullScalar(arrow::float64())};
  }

  switch (agg_op) {
    case OpType::AGG_SUM: {
      double sum = 0.0;
      bool any_valid = false;
      for (const auto& v : vals) {
        if (v && v->is_valid) { sum += scalar_to_double(v); any_valid = true; }
      }
      auto s = any_valid ? make_double(sum) : arrow::MakeNullScalar(arrow::float64());
      return std::vector<std::shared_ptr<arrow::Scalar>>(vals.size(), s);
    }
    case OpType::AGG_MEAN: {
      double sum = 0.0; int64_t cnt = 0;
      for (const auto& v : vals) {
        if (v && v->is_valid) { sum += scalar_to_double(v); cnt++; }
      }
      auto s = cnt > 0 ? make_double(sum / cnt) : arrow::MakeNullScalar(arrow::float64());
      return std::vector<std::shared_ptr<arrow::Scalar>>(vals.size(), s);
    }
    case OpType::AGG_COUNT: {
      int64_t cnt = 0;
      for (const auto& v : vals) if (v && v->is_valid) cnt++;
      auto s = make_int64(cnt);
      return std::vector<std::shared_ptr<arrow::Scalar>>(vals.size(), s);
    }
    case OpType::AGG_MIN: {
      std::shared_ptr<arrow::Scalar> mn;
      for (const auto& v : vals) {
        if (v && v->is_valid) {
          if (!mn || compare_scalars(v, mn) < 0) mn = v;
        }
      }
      if (!mn) mn = arrow::MakeNullScalar(arrow::float64());
      return std::vector<std::shared_ptr<arrow::Scalar>>(vals.size(), mn);
    }
    case OpType::AGG_MAX: {
      std::shared_ptr<arrow::Scalar> mx;
      for (const auto& v : vals) {
        if (v && v->is_valid) {
          if (!mx || compare_scalars(v, mx) > 0) mx = v;
        }
      }
      if (!mx) mx = arrow::MakeNullScalar(arrow::float64());
      return std::vector<std::shared_ptr<arrow::Scalar>>(vals.size(), mx);
    }
    default:
      return vals;
  }
}

}  // namespace dataframelib
