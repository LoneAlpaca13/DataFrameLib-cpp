#pragma once

#include <arrow/api.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace dataframelib {

// ================= BASE =================
class Expr {
 public:
  virtual ~Expr() = default;

  virtual std::vector<std::shared_ptr<arrow::Scalar>> evaluate(
      const std::shared_ptr<arrow::Table>& table) const = 0;
};

class AliasExpr : public Expr {
 private:
  std::shared_ptr<Expr> child;
  std::string alias;

 public:
  AliasExpr(std::shared_ptr<Expr> c, const std::string& name)
      : child(c), alias(name) {}

  std::vector<std::shared_ptr<arrow::Scalar>> evaluate(
      const std::shared_ptr<arrow::Table>& table) const override {
    return child->evaluate(table);
  }

  const std::string& getAlias() const { return alias; }
};

inline std::shared_ptr<Expr> alias(std::shared_ptr<Expr> e,
                                   const std::string& name) {
  return std::make_shared<AliasExpr>(e, name);
}

// ================= COLUMN =================
class ColumnExpr : public Expr {
 private:
  std::string name;

 public:
  ColumnExpr(const std::string& col) : name(col) {}

  std::pair<std::string, std::string> sum() const { return {name, "sum"}; }
  std::pair<std::string, std::string> mean() const { return {name, "mean"}; }
  std::pair<std::string, std::string> count() const { return {name, "count"}; }
  std::pair<std::string, std::string> min() const { return {name, "min"}; }
  std::pair<std::string, std::string> max() const { return {name, "max"}; }

  const std::string& getName() const { return name; }

  std::vector<std::shared_ptr<arrow::Scalar>> evaluate(
      const std::shared_ptr<arrow::Table>& table) const override {
    int idx = table->schema()->GetFieldIndex(name);
    if (idx == -1) throw std::runtime_error("Column not found: " + name);

    auto chunk = table->column(idx)->chunk(0);

    std::vector<std::shared_ptr<arrow::Scalar>> result;
    result.reserve(chunk->length());

    for (int i = 0; i < chunk->length(); i++) {
      result.push_back(chunk->GetScalar(i).ValueOrDie());
    }

    return result;
  }
};

// ================= LITERAL =================
class LiteralExpr : public Expr {
 private:
  std::shared_ptr<arrow::Scalar> value;

 public:
  LiteralExpr(std::shared_ptr<arrow::Scalar> val) : value(val) {}

  std::vector<std::shared_ptr<arrow::Scalar>> evaluate(
      const std::shared_ptr<arrow::Table>& table) const override {
    return std::vector<std::shared_ptr<arrow::Scalar>>(table->num_rows(),
                                                       value);
  }

  const std::shared_ptr<arrow::Scalar>& getValue() const { return value; }
};

// ================= OP TYPES =================
enum class OpType {
  ADD,
  SUB,
  MUL,
  DIV,
  MOD,

  GT,
  LT,
  GE,
  LE,
  EQ,
  NEQ,

  AND,
  OR,
  NOT,

  ABS,
  IS_NULL,
  IS_NOT_NULL,

  LENGTH,
  CONTAINS,
  STARTS_WITH,
  ENDS_WITH,
  TO_LOWER,
  TO_UPPER
};

// ================= HELPERS =================

inline std::shared_ptr<arrow::Scalar> make_null() {
  return std::make_shared<arrow::NullScalar>();
}

// ================= BINARY =================
class BinaryExpr : public Expr {
 private:
  std::shared_ptr<Expr> left;
  std::shared_ptr<Expr> right;
  OpType op;

 public:
  BinaryExpr(std::shared_ptr<Expr> l, std::shared_ptr<Expr> r, OpType o)
      : left(l), right(r), op(o) {}

  const std::shared_ptr<Expr>& getLeft() const { return left; }
  const std::shared_ptr<Expr>& getRight() const { return right; }
  OpType getOp() const { return op; }

  std::vector<std::shared_ptr<arrow::Scalar>> evaluate(
      const std::shared_ptr<arrow::Table>& table) const override {
    auto L = left->evaluate(table);
    auto R = right->evaluate(table);

    std::vector<std::shared_ptr<arrow::Scalar>> result;
    result.reserve(L.size());

    for (size_t i = 0; i < L.size(); i++) {
      auto l = L[i];
      auto r = R[i];

      // NULL PROPAGATION
      if (!l->is_valid || !r->is_valid) {
        result.push_back(make_null());
        continue;
      }
      // int + double → double
      if (std::dynamic_pointer_cast<arrow::Int64Scalar>(l) &&
          std::dynamic_pointer_cast<arrow::DoubleScalar>(r)) {
        double a = std::dynamic_pointer_cast<arrow::Int64Scalar>(l)->value;
        double b = std::dynamic_pointer_cast<arrow::DoubleScalar>(r)->value;

        result.push_back(std::make_shared<arrow::DoubleScalar>(a + b));
        continue;
      }

      // float + double → double
      if (std::dynamic_pointer_cast<arrow::FloatScalar>(l) &&
          std::dynamic_pointer_cast<arrow::DoubleScalar>(r)) {
        float a = std::dynamic_pointer_cast<arrow::FloatScalar>(l)->value;
        double b = std::dynamic_pointer_cast<arrow::DoubleScalar>(r)->value;

        result.push_back(std::make_shared<arrow::DoubleScalar>(a + b));
        continue;
      }

      // double + float → double
      if (std::dynamic_pointer_cast<arrow::DoubleScalar>(l) &&
          std::dynamic_pointer_cast<arrow::FloatScalar>(r)) {
        double a = std::dynamic_pointer_cast<arrow::DoubleScalar>(l)->value;
        float b = std::dynamic_pointer_cast<arrow::FloatScalar>(r)->value;

        result.push_back(std::make_shared<arrow::DoubleScalar>(a + b));
        continue;
      }

      // ===== INT =====
      if (auto li = std::dynamic_pointer_cast<arrow::Int64Scalar>(l)) {
        auto ri = std::dynamic_pointer_cast<arrow::Int64Scalar>(r);
        if (!ri) throw std::runtime_error("Type mismatch");

        int64_t a = li->value;
        int64_t b = ri->value;

        switch (op) {
          case OpType::ADD:
            result.push_back(std::make_shared<arrow::Int64Scalar>(a + b));
            break;
          case OpType::SUB:
            result.push_back(std::make_shared<arrow::Int64Scalar>(a - b));
            break;
          case OpType::MUL:
            result.push_back(std::make_shared<arrow::Int64Scalar>(a * b));
            break;
          case OpType::DIV:
            result.push_back(b == 0
                                 ? make_null()
                                 : std::make_shared<arrow::Int64Scalar>(a / b));
            break;
          case OpType::MOD:
            result.push_back(b == 0
                                 ? make_null()
                                 : std::make_shared<arrow::Int64Scalar>(a % b));
            break;

          case OpType::GT:
            result.push_back(std::make_shared<arrow::BooleanScalar>(a > b));
            break;
          case OpType::LT:
            result.push_back(std::make_shared<arrow::BooleanScalar>(a < b));
            break;
          case OpType::GE:
            result.push_back(std::make_shared<arrow::BooleanScalar>(a >= b));
            break;
          case OpType::LE:
            result.push_back(std::make_shared<arrow::BooleanScalar>(a <= b));
            break;
          case OpType::EQ:
            result.push_back(std::make_shared<arrow::BooleanScalar>(a == b));
            break;
          case OpType::NEQ:
            result.push_back(std::make_shared<arrow::BooleanScalar>(a != b));
            break;

          default:
            throw std::runtime_error("Invalid int op");
        }
        continue;
      }

      // ===== FLOAT (float32) =====
      if (auto lf = std::dynamic_pointer_cast<arrow::FloatScalar>(l)) {
        auto rf = std::dynamic_pointer_cast<arrow::FloatScalar>(r);
        if (!rf) throw std::runtime_error("Type mismatch");

        float a = lf->value;
        float b = rf->value;

        if (op == OpType::ADD)
          result.push_back(std::make_shared<arrow::FloatScalar>(a + b));
        else if (op == OpType::SUB)
          result.push_back(std::make_shared<arrow::FloatScalar>(a - b));
        else if (op == OpType::MUL)
          result.push_back(std::make_shared<arrow::FloatScalar>(a * b));
        else if (op == OpType::DIV)
          result.push_back(b == 0
                               ? make_null()
                               : std::make_shared<arrow::FloatScalar>(a / b));
        else
          throw std::runtime_error("Invalid float op");

        continue;
      }
      // ===== DOUBLE =====
      if (auto lf = std::dynamic_pointer_cast<arrow::DoubleScalar>(l)) {
        auto rf = std::dynamic_pointer_cast<arrow::DoubleScalar>(r);
        if (!rf) throw std::runtime_error("Type mismatch");

        double a = lf->value;
        double b = rf->value;

        switch (op) {
          case OpType::ADD:
            result.push_back(std::make_shared<arrow::DoubleScalar>(a + b));
            break;
          case OpType::SUB:
            result.push_back(std::make_shared<arrow::DoubleScalar>(a - b));
            break;
          case OpType::MUL:
            result.push_back(std::make_shared<arrow::DoubleScalar>(a * b));
            break;
          case OpType::DIV:
            result.push_back(
                b == 0 ? make_null()
                       : std::make_shared<arrow::DoubleScalar>(a / b));
            break;
          default:
            throw std::runtime_error("Invalid double op");
        }

        continue;
      }
      // ===== BOOL =====
      if (auto lb = std::dynamic_pointer_cast<arrow::BooleanScalar>(l)) {
        auto rb = std::dynamic_pointer_cast<arrow::BooleanScalar>(r);
        if (!rb) throw std::runtime_error("Type mismatch");

        if (op == OpType::AND)
          result.push_back(
              std::make_shared<arrow::BooleanScalar>(lb->value && rb->value));
        else if (op == OpType::OR)
          result.push_back(
              std::make_shared<arrow::BooleanScalar>(lb->value || rb->value));
        else if (op == OpType::EQ)
          result.push_back(
              std::make_shared<arrow::BooleanScalar>(lb->value == rb->value));
        else if (op == OpType::NEQ)
          result.push_back(
              std::make_shared<arrow::BooleanScalar>(lb->value != rb->value));
        else
          throw std::runtime_error("Invalid bool op");

        continue;
      }

      // ===== STRING =====
      if (auto ls = std::dynamic_pointer_cast<arrow::StringScalar>(l)) {
        auto rs = std::dynamic_pointer_cast<arrow::StringScalar>(r);
        if (!rs) throw std::runtime_error("Type mismatch");

        std::string a = ls->ToString();
        std::string b = rs->ToString();

        if (op == OpType::CONTAINS)
          result.push_back(std::make_shared<arrow::BooleanScalar>(
              a.find(b) != std::string::npos));
        else if (op == OpType::STARTS_WITH)
          result.push_back(
              std::make_shared<arrow::BooleanScalar>(a.rfind(b, 0) == 0));
        else if (op == OpType::ENDS_WITH)
          result.push_back(std::make_shared<arrow::BooleanScalar>(
              a.size() >= b.size() &&
              a.compare(a.size() - b.size(), b.size(), b) == 0));
        else if (op == OpType::EQ)
          result.push_back(std::make_shared<arrow::BooleanScalar>(a == b));
        else if (op == OpType::NEQ)
          result.push_back(std::make_shared<arrow::BooleanScalar>(a != b));
        else
          throw std::runtime_error("Invalid string op");

        continue;
      }

      throw std::runtime_error("Unsupported type");
    }

    return result;
  }
};

// ================= UNARY =================
class UnaryExpr : public Expr {
 private:
  std::shared_ptr<Expr> child;
  OpType op;

 public:
  UnaryExpr(std::shared_ptr<Expr> c, OpType o) : child(c), op(o) {}

  std::vector<std::shared_ptr<arrow::Scalar>> evaluate(
      const std::shared_ptr<arrow::Table>& table) const override {
    auto vals = child->evaluate(table);
    std::vector<std::shared_ptr<arrow::Scalar>> result;

    for (auto& v : vals) {
      if (!v->is_valid) {
        if (op == OpType::IS_NULL)
          result.push_back(std::make_shared<arrow::BooleanScalar>(true));
        else if (op == OpType::IS_NOT_NULL)
          result.push_back(std::make_shared<arrow::BooleanScalar>(false));
        else
          result.push_back(make_null());
        continue;
      }

      if (auto i = std::dynamic_pointer_cast<arrow::Int64Scalar>(v)) {
        if (op == OpType::ABS)
          result.push_back(
              std::make_shared<arrow::Int64Scalar>(std::abs(i->value)));
        else
          throw std::runtime_error("Invalid int unary");
        continue;
      }

      if (auto b = std::dynamic_pointer_cast<arrow::BooleanScalar>(v)) {
        if (op == OpType::NOT)
          result.push_back(std::make_shared<arrow::BooleanScalar>(!b->value));
        else
          throw std::runtime_error("Invalid bool unary");
        continue;
      }

      if (auto s = std::dynamic_pointer_cast<arrow::StringScalar>(v)) {
        std::string str = s->ToString();

        if (op == OpType::LENGTH)
          result.push_back(
              std::make_shared<arrow::Int64Scalar>((int64_t)str.size()));
        else if (op == OpType::TO_LOWER) {
          std::transform(str.begin(), str.end(), str.begin(), ::tolower);
          result.push_back(std::make_shared<arrow::StringScalar>(str));
        } else if (op == OpType::TO_UPPER) {
          std::transform(str.begin(), str.end(), str.begin(), ::toupper);
          result.push_back(std::make_shared<arrow::StringScalar>(str));
        } else
          throw std::runtime_error("Invalid string unary");

        continue;
      }

      throw std::runtime_error("Unsupported unary type");
    }

    return result;
  }
};

// ================= HELPERS =================
inline std::shared_ptr<Expr> col(const std::string& name) {
  return std::make_shared<ColumnExpr>(name);
}

inline std::shared_ptr<Expr> operator~(std::shared_ptr<Expr> a) {
  return std::make_shared<UnaryExpr>(a, OpType::NOT);
}

inline std::pair<std::string, std::string> sum(std::shared_ptr<Expr> e) {
  auto c = std::dynamic_pointer_cast<ColumnExpr>(e);
  if (!c) throw std::runtime_error("sum() expects column");
  return {c->getName(), "sum"};
}

inline std::pair<std::string, std::string> mean(std::shared_ptr<Expr> e) {
  auto c = std::dynamic_pointer_cast<ColumnExpr>(e);
  if (!c) throw std::runtime_error("mean() expects column");
  return {c->getName(), "mean"};
}

inline std::pair<std::string, std::string> count(std::shared_ptr<Expr> e) {
  auto c = std::dynamic_pointer_cast<ColumnExpr>(e);
  if (!c) throw std::runtime_error("count() expects column");
  return {c->getName(), "count"};
}

inline std::pair<std::string, std::string> min(std::shared_ptr<Expr> e) {
  auto c = std::dynamic_pointer_cast<ColumnExpr>(e);
  if (!c) throw std::runtime_error("min() expects column");
  return {c->getName(), "min"};
}

inline std::pair<std::string, std::string> max(std::shared_ptr<Expr> e) {
  auto c = std::dynamic_pointer_cast<ColumnExpr>(e);
  if (!c) throw std::runtime_error("max() expects column");
  return {c->getName(), "max"};
}

inline std::shared_ptr<Expr> lit(int64_t v) {
  return std::make_shared<LiteralExpr>(std::make_shared<arrow::Int64Scalar>(v));
}

inline std::pair<std::string, std::string> sum(const std::string& col) {
  return {col, "sum"};
}

inline std::pair<std::string, std::string> mean(const std::string& col) {
  return {col, "mean"};
}

inline std::pair<std::string, std::string> count(const std::string& col) {
  return {col, "count"};
}

inline std::pair<std::string, std::string> min(const std::string& col) {
  return {col, "min"};
}

inline std::pair<std::string, std::string> max(const std::string& col) {
  return {col, "max"};
}

inline std::shared_ptr<Expr> is_null(std::shared_ptr<Expr> a) {
  return std::make_shared<UnaryExpr>(a, OpType::IS_NULL);
}

inline std::shared_ptr<Expr> is_not_null(std::shared_ptr<Expr> a) {
  return std::make_shared<UnaryExpr>(a, OpType::IS_NOT_NULL);
}

inline std::shared_ptr<Expr> operator%(std::shared_ptr<Expr> a,
                                       std::shared_ptr<Expr> b) {
  return std::make_shared<BinaryExpr>(a, b, OpType::MOD);
}

inline std::shared_ptr<Expr> abs(std::shared_ptr<Expr> a) {
  return std::make_shared<UnaryExpr>(a, OpType::ABS);
}

// ================= OPERATORS =================
inline std::shared_ptr<Expr> operator+(std::shared_ptr<Expr> a,
                                       std::shared_ptr<Expr> b) {
  return std::make_shared<BinaryExpr>(a, b, OpType::ADD);
}
inline std::shared_ptr<Expr> operator-(std::shared_ptr<Expr> a,
                                       std::shared_ptr<Expr> b) {
  return std::make_shared<BinaryExpr>(a, b, OpType::SUB);
}
inline std::shared_ptr<Expr> operator*(std::shared_ptr<Expr> a,
                                       std::shared_ptr<Expr> b) {
  return std::make_shared<BinaryExpr>(a, b, OpType::MUL);
}
inline std::shared_ptr<Expr> operator/(std::shared_ptr<Expr> a,
                                       std::shared_ptr<Expr> b) {
  return std::make_shared<BinaryExpr>(a, b, OpType::DIV);
}
inline std::shared_ptr<Expr> operator>(std::shared_ptr<Expr> a,
                                       std::shared_ptr<Expr> b) {
  return std::make_shared<BinaryExpr>(a, b, OpType::GT);
}
inline std::shared_ptr<Expr> operator<(std::shared_ptr<Expr> a,
                                       std::shared_ptr<Expr> b) {
  return std::make_shared<BinaryExpr>(a, b, OpType::LT);
}
inline std::shared_ptr<Expr> operator==(std::shared_ptr<Expr> a,
                                        std::shared_ptr<Expr> b) {
  return std::make_shared<BinaryExpr>(a, b, OpType::EQ);
}
inline std::shared_ptr<Expr> operator!=(std::shared_ptr<Expr> a,
                                        std::shared_ptr<Expr> b) {
  return std::make_shared<BinaryExpr>(a, b, OpType::NEQ);
}
inline std::shared_ptr<Expr> operator&&(std::shared_ptr<Expr> a,
                                        std::shared_ptr<Expr> b) {
  return std::make_shared<BinaryExpr>(a, b, OpType::AND);
}
inline std::shared_ptr<Expr> operator||(std::shared_ptr<Expr> a,
                                        std::shared_ptr<Expr> b) {
  return std::make_shared<BinaryExpr>(a, b, OpType::OR);
}

// ================= STRING HELPERS =================

inline std::shared_ptr<Expr> length(std::shared_ptr<Expr> a) {
  return std::make_shared<UnaryExpr>(a, OpType::LENGTH);
}

inline std::shared_ptr<Expr> contains(std::shared_ptr<Expr> a,
                                      std::shared_ptr<Expr> b) {
  return std::make_shared<BinaryExpr>(a, b, OpType::CONTAINS);
}

inline std::shared_ptr<Expr> starts_with(std::shared_ptr<Expr> a,
                                         std::shared_ptr<Expr> b) {
  return std::make_shared<BinaryExpr>(a, b, OpType::STARTS_WITH);
}

inline std::shared_ptr<Expr> ends_with(std::shared_ptr<Expr> a,
                                       std::shared_ptr<Expr> b) {
  return std::make_shared<BinaryExpr>(a, b, OpType::ENDS_WITH);
}

inline std::shared_ptr<Expr> to_lower(std::shared_ptr<Expr> a) {
  return std::make_shared<UnaryExpr>(a, OpType::TO_LOWER);
}

inline std::shared_ptr<Expr> to_upper(std::shared_ptr<Expr> a) {
  return std::make_shared<UnaryExpr>(a, OpType::TO_UPPER);
}
}  // namespace dataframelib