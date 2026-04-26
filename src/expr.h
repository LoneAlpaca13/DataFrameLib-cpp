#pragma once
#include <arrow/api.h>

#include <cmath>
#include <memory>
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
      auto scalar = chunk->GetScalar(i).ValueOrDie();

      if (!scalar->is_valid) {
        result.push_back(scalar);  // keep null
      } else {
        result.push_back(scalar);
      }
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

    std::vector<std::shared_ptr<arrow::Scalar>> result;
    result.reserve(rows);

    for (int i = 0; i < rows; i++) {
      result.push_back(value);
    }

    return result;
  }
};

// ================= OP ENUM =================
enum class OpType { ADD, SUB, MUL, DIV, MOD, GT, LT, GE, LE, EQ, NEQ, AND, OR };

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
      auto l = lvals[i];
      auto r = rvals[i];

      // ================= NULL PROPAGATION =================
      if (!l->is_valid || !r->is_valid) {
        result.push_back(arrow::MakeNullScalar(arrow::null()));
        continue;
      }

      // ================= INT =================
      auto l_int = std::dynamic_pointer_cast<arrow::Int64Scalar>(l);
      auto r_int = std::dynamic_pointer_cast<arrow::Int64Scalar>(r);

      // ================= FLOAT =================
      auto l_float = std::dynamic_pointer_cast<arrow::DoubleScalar>(l);
      auto r_float = std::dynamic_pointer_cast<arrow::DoubleScalar>(r);

      // ================= TYPE PROMOTION =================
      // int + float → float
      if ((l_int && r_float) || (l_float && r_int) || (l_float && r_float)) {
        double lv = l_float ? l_float->value : l_int->value;
        double rv = r_float ? r_float->value : r_int->value;

        switch (op) {
          case OpType::ADD:
            result.push_back(std::make_shared<arrow::DoubleScalar>(lv + rv));
            break;
          case OpType::SUB:
            result.push_back(std::make_shared<arrow::DoubleScalar>(lv - rv));
            break;
          case OpType::MUL:
            result.push_back(std::make_shared<arrow::DoubleScalar>(lv * rv));
            break;
          case OpType::DIV:
            if (rv == 0) {
              result.push_back(arrow::MakeNullScalar(arrow::float64()));
            } else {
              result.push_back(std::make_shared<arrow::DoubleScalar>(lv / rv));
            }
            break;
          case OpType::MOD:
            result.push_back(
                std::make_shared<arrow::DoubleScalar>(fmod(lv, rv)));
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
            throw std::runtime_error("Invalid float operation");
        }

        continue;
      }

      // ================= PURE INT =================
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
            if (rv == 0) {
              result.push_back(arrow::MakeNullScalar(arrow::int64()));
            } else {
              result.push_back(std::make_shared<arrow::Int64Scalar>(lv / rv));
            }
            break;
          case OpType::MOD:
            if (rv == 0) {
              result.push_back(arrow::MakeNullScalar(arrow::int64()));
            } else {
              result.push_back(std::make_shared<arrow::Int64Scalar>(lv % rv));
            }
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

      // ================= BOOLEAN =================
      auto l_bool = std::dynamic_pointer_cast<arrow::BooleanScalar>(l);
      auto r_bool = std::dynamic_pointer_cast<arrow::BooleanScalar>(r);

      if (l_bool && r_bool) {
        bool lv = l_bool->value;
        bool rv = r_bool->value;

        switch (op) {
          case OpType::AND:
            result.push_back(std::make_shared<arrow::BooleanScalar>(lv && rv));
            break;
          case OpType::OR:
            result.push_back(std::make_shared<arrow::BooleanScalar>(lv || rv));
            break;
          default:
            throw std::runtime_error("Invalid boolean operation");
        }

        continue;
      }

      throw std::runtime_error("Type mismatch in BinaryExpr");
    }

    return result;
  }
};
}  // namespace dataframelib