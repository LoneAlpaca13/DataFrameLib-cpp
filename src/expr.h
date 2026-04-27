#pragma once

#include <arrow/api.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace dataframelib {

// ================= FORWARD =================
class Expr;
std::shared_ptr<Expr> lit(const std::string& value);

// ================= BASE =================
class Expr {
 public:
  virtual ~Expr() = default;

  virtual std::vector<std::shared_ptr<arrow::Scalar>> evaluate(
      const std::shared_ptr<arrow::Table>& table) const = 0;
};

// ================= COLUMN =================
class ColumnExpr : public Expr {
 private:
  std::string name;

 public:
  ColumnExpr(const std::string& col) : name(col) {}

  const std::string& getName() const { return name; }

  std::vector<std::shared_ptr<arrow::Scalar>> evaluate(
      const std::shared_ptr<arrow::Table>& table) const override {
    int idx = table->schema()->GetFieldIndex(name);
    if (idx == -1) throw std::runtime_error("Column not found");

    auto arr = table->column(idx)->chunk(0);

    std::vector<std::shared_ptr<arrow::Scalar>> out;
    for (int i = 0; i < arr->length(); i++) {
      out.push_back(arr->GetScalar(i).ValueOrDie());
    }
    return out;
  }
};

// ================= LITERAL =================
class LiteralExpr : public Expr {
 private:
  std::shared_ptr<arrow::Scalar> value;

 public:
  LiteralExpr(std::shared_ptr<arrow::Scalar> v) : value(v) {}

  const std::shared_ptr<arrow::Scalar>& getValue() const { return value; }

  std::vector<std::shared_ptr<arrow::Scalar>> evaluate(
      const std::shared_ptr<arrow::Table>& table) const override {
    return std::vector<std::shared_ptr<arrow::Scalar>>(table->num_rows(),
                                                       value);
  }
};

// ================= OPS =================
enum class OpType {
  ADD,
  SUB,
  MUL,
  DIV,
  EQ,
  NEQ,
  GT,
  LT,
  AND,
  OR,
  LENGTH,
  CONTAINS
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

  std::shared_ptr<Expr> getLeft() const { return left; }
  std::shared_ptr<Expr> getRight() const { return right; }
  OpType getOp() const { return op; }

  std::vector<std::shared_ptr<arrow::Scalar>> evaluate(
      const std::shared_ptr<arrow::Table>& table) const override {
    auto L = left->evaluate(table);
    auto R = right->evaluate(table);

    std::vector<std::shared_ptr<arrow::Scalar>> result;

    for (size_t i = 0; i < L.size(); i++) {
      auto l = L[i];
      auto r = R[i];

      if (!l->is_valid || !r->is_valid) {
        result.push_back(make_null());
        continue;
      }

      // ===== INT =====
      if (auto li = std::dynamic_pointer_cast<arrow::Int64Scalar>(l)) {
        auto ri = std::dynamic_pointer_cast<arrow::Int64Scalar>(r);
        if (!ri) throw std::runtime_error("Type mismatch");

        int64_t a = li->value;
        int64_t b = ri->value;

        if (op == OpType::ADD)
          result.push_back(std::make_shared<arrow::Int64Scalar>(a + b));
        else if (op == OpType::EQ)
          result.push_back(std::make_shared<arrow::BooleanScalar>(a == b));
        else if (op == OpType::NEQ)
          result.push_back(std::make_shared<arrow::BooleanScalar>(a != b));
        else
          throw std::runtime_error("Unsupported int op");

        continue;
      }

      // ===== STRING =====
      if (auto ls = std::dynamic_pointer_cast<arrow::StringScalar>(l)) {
        auto rs = std::dynamic_pointer_cast<arrow::StringScalar>(r);
        if (!rs) throw std::runtime_error("Type mismatch");

        std::string a = ls->value->ToString();
        std::string b = rs->value->ToString();

        if (op == OpType::EQ)
          result.push_back(std::make_shared<arrow::BooleanScalar>(a == b));
        else if (op == OpType::NEQ)
          result.push_back(std::make_shared<arrow::BooleanScalar>(a != b));
        else if (op == OpType::CONTAINS)
          result.push_back(std::make_shared<arrow::BooleanScalar>(
              a.find(b) != std::string::npos));
        else
          throw std::runtime_error("Unsupported string op");

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
        result.push_back(make_null());
        continue;
      }

      if (auto s = std::dynamic_pointer_cast<arrow::StringScalar>(v)) {
        std::string str = s->value->ToString();

        if (op == OpType::LENGTH)
          result.push_back(
              std::make_shared<arrow::Int64Scalar>((int64_t)str.size()));
        else
          throw std::runtime_error("Unsupported unary op");

        continue;
      }

      throw std::runtime_error("Unsupported unary");
    }

    return result;
  }
};

// ================= BUILDERS =================
inline std::shared_ptr<Expr> col(const std::string& name) {
  return std::make_shared<ColumnExpr>(name);
}

// ===== LITERALS =====
inline std::shared_ptr<Expr> lit(int64_t v) {
  return std::make_shared<LiteralExpr>(std::make_shared<arrow::Int64Scalar>(v));
}

inline std::shared_ptr<Expr> lit(const std::string& s) {
  return std::make_shared<LiteralExpr>(
      std::make_shared<arrow::StringScalar>(s));
}

// ===== STRING OPERATORS =====
inline std::shared_ptr<Expr> operator==(std::shared_ptr<Expr> lhs,
                                        const std::string& rhs) {
  return std::make_shared<BinaryExpr>(lhs, lit(rhs), OpType::EQ);
}

inline std::shared_ptr<Expr> operator!=(std::shared_ptr<Expr> lhs,
                                        const std::string& rhs) {
  return std::make_shared<BinaryExpr>(lhs, lit(rhs), OpType::NEQ);
}

inline std::shared_ptr<Expr> operator==(std::shared_ptr<Expr> lhs,
                                        const char* rhs) {
  return lhs == std::string(rhs);
}

inline std::shared_ptr<Expr> operator!=(std::shared_ptr<Expr> lhs,
                                        const char* rhs) {
  return lhs != std::string(rhs);
}
inline std::shared_ptr<Expr> operator==(const std::string& lhs,
                                        std::shared_ptr<Expr> rhs) {
  return std::make_shared<BinaryExpr>(lit(lhs), rhs, OpType::EQ);
}

inline std::shared_ptr<Expr> operator!=(const std::string& lhs,
                                        std::shared_ptr<Expr> rhs) {
  return std::make_shared<BinaryExpr>(lit(lhs), rhs, OpType::NEQ);
}

inline std::shared_ptr<Expr> operator==(const char* lhs,
                                        std::shared_ptr<Expr> rhs) {
  return std::string(lhs) == rhs;
}

inline std::shared_ptr<Expr> operator!=(const char* lhs,
                                        std::shared_ptr<Expr> rhs) {
  return std::string(lhs) != rhs;
}
// ===== OTHER OPERATORS =====
inline std::shared_ptr<Expr> operator+(std::shared_ptr<Expr> a,
                                       std::shared_ptr<Expr> b) {
  return std::make_shared<BinaryExpr>(a, b, OpType::ADD);
}

inline std::shared_ptr<Expr> operator==(std::shared_ptr<Expr> a,
                                        std::shared_ptr<Expr> b) {
  return std::make_shared<BinaryExpr>(a, b, OpType::EQ);
}
inline std::shared_ptr<Expr> lit(const char* s) { return lit(std::string(s)); }

inline std::shared_ptr<Expr> operator!=(std::shared_ptr<Expr> a,
                                        std::shared_ptr<Expr> b) {
  return std::make_shared<BinaryExpr>(a, b, OpType::NEQ);
}

// ===== STRING FUNCS =====
inline std::shared_ptr<Expr> length(std::shared_ptr<Expr> a) {
  return std::make_shared<UnaryExpr>(a, OpType::LENGTH);
}

inline std::shared_ptr<Expr> contains(std::shared_ptr<Expr> a,
                                      std::shared_ptr<Expr> b) {
  return std::make_shared<BinaryExpr>(a, b, OpType::CONTAINS);
}

}  // namespace dataframelib