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
  ABS,
  IS_NULL,
  IS_NOT_NULL,
  NOT,
  LENGTH,
  CONTAINS,
  STARTS_WITH,
  ENDS_WITH,
  TO_LOWER,
  TO_UPPER
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

  // ✅ REQUIRED FOR OPTIMIZER
  const std::shared_ptr<Expr>& getLeft() const { return left; }
  const std::shared_ptr<Expr>& getRight() const { return right; }
  OpType getOp() const { return op; }

  std::vector<std::shared_ptr<arrow::Scalar>> evaluate(
      const std::shared_ptr<arrow::Table>& table) const override {
    auto lvals = left->evaluate(table);
    auto rvals = right->evaluate(table);

    std::vector<std::shared_ptr<arrow::Scalar>> result;
    result.reserve(lvals.size());

    for (size_t i = 0; i < lvals.size(); i++) {
      auto L = lvals[i];
      auto R = rvals[i];

      if (!L->is_valid || !R->is_valid) {
        result.push_back(std::make_shared<arrow::NullScalar>());
        continue;
      }

      // ===== INT =====
      auto li = std::dynamic_pointer_cast<arrow::Int64Scalar>(L);
      auto ri = std::dynamic_pointer_cast<arrow::Int64Scalar>(R);

      if (li && ri) {
        int64_t lv = li->value;
        int64_t rv = ri->value;

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
      auto lb = std::dynamic_pointer_cast<arrow::BooleanScalar>(L);
      auto rb = std::dynamic_pointer_cast<arrow::BooleanScalar>(R);

      if (lb && rb) {
        if (op == OpType::AND)
          result.push_back(
              std::make_shared<arrow::BooleanScalar>(lb->value && rb->value));
        else if (op == OpType::OR)
          result.push_back(
              std::make_shared<arrow::BooleanScalar>(lb->value || rb->value));
        else
          throw std::runtime_error("Invalid boolean operation");

        continue;
      }

      // ===== STRING =====
      auto ls = std::dynamic_pointer_cast<arrow::StringScalar>(L);
      auto rs = std::dynamic_pointer_cast<arrow::StringScalar>(R);

      if (ls && rs) {
        std::string lv = ls->ToString();
        std::string rv = rs->ToString();

        if (op == OpType::CONTAINS)
          result.push_back(std::make_shared<arrow::BooleanScalar>(
              lv.find(rv) != std::string::npos));

        else if (op == OpType::STARTS_WITH)
          result.push_back(
              std::make_shared<arrow::BooleanScalar>(lv.rfind(rv, 0) == 0));

        else if (op == OpType::ENDS_WITH)
          result.push_back(std::make_shared<arrow::BooleanScalar>(
              lv.size() >= rv.size() &&
              lv.compare(lv.size() - rv.size(), rv.size(), rv) == 0));

        else
          throw std::runtime_error("Invalid string operation");

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
        }

        else if (op == OpType::TO_UPPER) {
          std::transform(str.begin(), str.end(), str.begin(), ::toupper);
          result.push_back(std::make_shared<arrow::StringScalar>(str));
        }

        else
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

inline std::shared_ptr<Expr> lit(std::shared_ptr<arrow::Scalar> v) {
  return std::make_shared<LiteralExpr>(v);
}

inline std::shared_ptr<Expr> abs(std::shared_ptr<Expr> e) {
  return std::make_shared<UnaryExpr>(e, OpType::ABS);
}

inline std::shared_ptr<Expr> length(std::shared_ptr<Expr> e) {
  return std::make_shared<UnaryExpr>(e, OpType::LENGTH);
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

inline std::shared_ptr<Expr> to_lower(std::shared_ptr<Expr> e) {
  return std::make_shared<UnaryExpr>(e, OpType::TO_LOWER);
}

inline std::shared_ptr<Expr> to_upper(std::shared_ptr<Expr> e) {
  return std::make_shared<UnaryExpr>(e, OpType::TO_UPPER);
}

}  // namespace dataframelib