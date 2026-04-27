#pragma once

#include <arrow/api.h>

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
    if (idx == -1) {
      throw std::runtime_error("Column not found: " + name);
    }

    auto column = table->column(idx);
    auto chunk = column->chunk(0);

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
    int rows = table->num_rows();

    std::vector<std::shared_ptr<arrow::Scalar>> result(rows, value);
    return result;
  }
};

// ================= OP ENUM =================

enum class OpType {
  // arithmetic
  ADD,
  SUB,
  MUL,
  DIV,
  MOD,

  // comparison
  GT,
  LT,
  GE,
  LE,
  EQ,
  NEQ,

  // logical
  AND,
  OR,

  // unary
  ABS,
  IS_NULL,
  IS_NOT_NULL,
  NOT,

  // string
  LENGTH,
  CONTAINS
};

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

  std::vector<std::shared_ptr<arrow::Scalar>> evaluate(
      const std::shared_ptr<arrow::Table>& table) const override {
    auto lvals = left->evaluate(table);
    auto rvals = right->evaluate(table);

    std::vector<std::shared_ptr<arrow::Scalar>> result;
    result.reserve(lvals.size());

    for (size_t i = 0; i < lvals.size(); i++) {
      auto L = lvals[i];
      auto R = rvals[i];

      // NULL propagation
      if (!L->is_valid || !R->is_valid) {
        result.push_back(std::make_shared<arrow::NullScalar>());
        continue;
      }

      // ===== INT =====
      auto l_int = std::dynamic_pointer_cast<arrow::Int64Scalar>(L);
      auto r_int = std::dynamic_pointer_cast<arrow::Int64Scalar>(R);

      if (l_int && r_int) {
        int64_t lv = l_int->value;
        int64_t rv = r_int->value;

        switch (op) {
          case OpType::ADD:
            result.push_back(std::make_shared<arrow::Int64Scalar>(lv + rv));
            break;
          case OpType::SUB:
            result.push_back(std::make_shared<arrow::Int64Scalar>(lv - rv));
            break;
          case OpType::MUL:
            result.push_back(std::make_shared<arrow::Int64Scalar>(lv * rv));
            break;
          case OpType::DIV:
            if (rv == 0)
              result.push_back(std::make_shared<arrow::NullScalar>());
            else
              result.push_back(std::make_shared<arrow::Int64Scalar>(lv / rv));
            break;
          case OpType::MOD:
            if (rv == 0)
              result.push_back(std::make_shared<arrow::NullScalar>());
            else
              result.push_back(std::make_shared<arrow::Int64Scalar>(lv % rv));
            break;

          case OpType::GT:
            result.push_back(std::make_shared<arrow::BooleanScalar>(lv > rv));
            break;
          case OpType::LT:
            result.push_back(std::make_shared<arrow::BooleanScalar>(lv < rv));
            break;
          case OpType::GE:
            result.push_back(std::make_shared<arrow::BooleanScalar>(lv >= rv));
            break;
          case OpType::LE:
            result.push_back(std::make_shared<arrow::BooleanScalar>(lv <= rv));
            break;
          case OpType::EQ:
            result.push_back(std::make_shared<arrow::BooleanScalar>(lv == rv));
            break;
          case OpType::NEQ:
            result.push_back(std::make_shared<arrow::BooleanScalar>(lv != rv));
            break;

          default:
            throw std::runtime_error("Invalid int operation");
        }
        continue;
      }

      // ===== BOOL =====
      auto l_bool = std::dynamic_pointer_cast<arrow::BooleanScalar>(L);
      auto r_bool = std::dynamic_pointer_cast<arrow::BooleanScalar>(R);

      if (l_bool && r_bool) {
        bool lv = l_bool->value;
        bool rv = r_bool->value;

        if (op == OpType::AND)
          result.push_back(std::make_shared<arrow::BooleanScalar>(lv && rv));
        else if (op == OpType::OR)
          result.push_back(std::make_shared<arrow::BooleanScalar>(lv || rv));
        else
          throw std::runtime_error("Invalid boolean operation");

        continue;
      }

      // ===== STRING =====
      auto l_str = std::dynamic_pointer_cast<arrow::StringScalar>(L);
      auto r_str = std::dynamic_pointer_cast<arrow::StringScalar>(R);

      if (l_str && r_str) {
        std::string lv = l_str->ToString();
        std::string rv = r_str->ToString();

        if (op == OpType::CONTAINS) {
          result.push_back(std::make_shared<arrow::BooleanScalar>(
              lv.find(rv) != std::string::npos));
        } else {
          throw std::runtime_error("Invalid string operation");
        }
        continue;
      }

      throw std::runtime_error("Type mismatch in BinaryExpr");
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
    result.reserve(vals.size());

    for (auto& v : vals) {
      if (!v->is_valid) {
        if (op == OpType::IS_NULL)
          result.push_back(std::make_shared<arrow::BooleanScalar>(true));
        else if (op == OpType::IS_NOT_NULL)
          result.push_back(std::make_shared<arrow::BooleanScalar>(false));
        else
          result.push_back(std::make_shared<arrow::NullScalar>());
        continue;
      }

      // INT
      if (auto i = std::dynamic_pointer_cast<arrow::Int64Scalar>(v)) {
        int64_t x = i->value;

        if (op == OpType::ABS)
          result.push_back(std::make_shared<arrow::Int64Scalar>(std::abs(x)));
        else if (op == OpType::IS_NULL)
          result.push_back(std::make_shared<arrow::BooleanScalar>(false));
        else if (op == OpType::IS_NOT_NULL)
          result.push_back(std::make_shared<arrow::BooleanScalar>(true));
        else
          throw std::runtime_error("Invalid unary int op");

        continue;
      }

      // BOOL
      if (auto b = std::dynamic_pointer_cast<arrow::BooleanScalar>(v)) {
        bool val = b->value;

        if (op == OpType::NOT)
          result.push_back(std::make_shared<arrow::BooleanScalar>(!val));
        else if (op == OpType::IS_NULL)
          result.push_back(std::make_shared<arrow::BooleanScalar>(false));
        else if (op == OpType::IS_NOT_NULL)
          result.push_back(std::make_shared<arrow::BooleanScalar>(true));
        else
          throw std::runtime_error("Invalid unary bool op");

        continue;
      }

      // STRING
      if (auto s = std::dynamic_pointer_cast<arrow::StringScalar>(v)) {
        std::string str = s->ToString();

        if (op == OpType::LENGTH)
          result.push_back(
              std::make_shared<arrow::Int64Scalar>((int64_t)str.size()));
        else if (op == OpType::IS_NULL)
          result.push_back(std::make_shared<arrow::BooleanScalar>(false));
        else if (op == OpType::IS_NOT_NULL)
          result.push_back(std::make_shared<arrow::BooleanScalar>(true));
        else
          throw std::runtime_error("Invalid unary string op");

        continue;
      }

      throw std::runtime_error("Unsupported type in UnaryExpr");
    }

    return result;
  }
};

// ================= HELPERS =================

// numeric
inline std::shared_ptr<Expr> abs(std::shared_ptr<Expr> e) {
  return std::make_shared<UnaryExpr>(e, OpType::ABS);
}

// null checks
inline std::shared_ptr<Expr> is_null(std::shared_ptr<Expr> e) {
  return std::make_shared<UnaryExpr>(e, OpType::IS_NULL);
}

inline std::shared_ptr<Expr> is_not_null(std::shared_ptr<Expr> e) {
  return std::make_shared<UnaryExpr>(e, OpType::IS_NOT_NULL);
}

// boolean
inline std::shared_ptr<Expr> not_(std::shared_ptr<Expr> e) {
  return std::make_shared<UnaryExpr>(e, OpType::NOT);
}

// string
inline std::shared_ptr<Expr> length(std::shared_ptr<Expr> e) {
  return std::make_shared<UnaryExpr>(e, OpType::LENGTH);
}

inline std::shared_ptr<Expr> contains(std::shared_ptr<Expr> a,
                                      std::shared_ptr<Expr> b) {
  return std::make_shared<BinaryExpr>(a, b, OpType::CONTAINS);
}

}  // namespace dataframelib