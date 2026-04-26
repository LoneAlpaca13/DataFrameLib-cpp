#pragma once
#include <arrow/api.h>

#include <memory>
#include <string>
#include <vector>

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

  std::vector<std::shared_ptr<arrow::Scalar>> evaluate(
      const std::shared_ptr<arrow::Table>& table) const override {
    int idx = table->schema()->GetFieldIndex(name);
    if (idx == -1) {
      throw std::runtime_error("Column not found: " + name);
    }

    auto column = table->column(idx);
    auto chunk = column->chunk(0);

    std::vector<std::shared_ptr<arrow::Scalar>> result;

    for (int i = 0; i < chunk->length(); i++) {
      result.push_back(chunk->GetScalar(i).ValueOrDie());
    }

    return result;
  }
};

class LiteralExpr : public Expr {
 private:
  std::shared_ptr<arrow::Scalar> value;

 public:
  LiteralExpr(std::shared_ptr<arrow::Scalar> val) : value(val) {}

  std::vector<std::shared_ptr<arrow::Scalar>> evaluate(
      const std::shared_ptr<arrow::Table>& table) const override {
    std::vector<std::shared_ptr<arrow::Scalar>> result;

    int rows = table->num_rows();
    for (int i = 0; i < rows; i++) {
      result.push_back(value);
    }

    return result;
  }
};

class GreaterExpr : public Expr {
 private:
  std::shared_ptr<Expr> left;
  std::shared_ptr<Expr> right;

 public:
  GreaterExpr(std::shared_ptr<Expr> l, std::shared_ptr<Expr> r)
      : left(l), right(r) {}

  std::vector<std::shared_ptr<arrow::Scalar>> evaluate(
      const std::shared_ptr<arrow::Table>& table) const override {
    auto lvals = left->evaluate(table);
    auto rvals = right->evaluate(table);

    std::vector<std::shared_ptr<arrow::Scalar>> result;

    for (size_t i = 0; i < lvals.size(); i++) {
      auto l = std::dynamic_pointer_cast<arrow::Int64Scalar>(lvals[i]);
      auto r = std::dynamic_pointer_cast<arrow::Int64Scalar>(rvals[i]);

      bool val = l->value > r->value;
      result.push_back(std::make_shared<arrow::BooleanScalar>(val));
    }

    return result;
  }
};

class AddExpr : public Expr {
 private:
  std::shared_ptr<Expr> left;
  std::shared_ptr<Expr> right;

 public:
  AddExpr(std::shared_ptr<Expr> l, std::shared_ptr<Expr> r)
      : left(l), right(r) {}

  std::vector<std::shared_ptr<arrow::Scalar>> evaluate(
      const std::shared_ptr<arrow::Table>& table) const override {
    auto lvals = left->evaluate(table);
    auto rvals = right->evaluate(table);

    std::vector<std::shared_ptr<arrow::Scalar>> result;

    for (size_t i = 0; i < lvals.size(); i++) {
      auto l = std::dynamic_pointer_cast<arrow::Int64Scalar>(lvals[i]);
      auto r = std::dynamic_pointer_cast<arrow::Int64Scalar>(rvals[i]);

      int64_t val = l->value + r->value;
      result.push_back(std::make_shared<arrow::Int64Scalar>(val));
    }

    return result;
  }
};

enum class OpType { ADD, GT, LT, GE, LE, EQ, NEQ, AND, OR };

class BinaryExpr : public Expr {
 private:
  std::shared_ptr<Expr> left;
  std::shared_ptr<Expr> right;
  OpType op;

 public:
  BinaryExpr(std::shared_ptr<Expr> l, std::shared_ptr<Expr> r, OpType o)
      : left(l), right(r), op(o) {}

  std::vector<std::shared_ptr<arrow::Scalar>> evaluate(
      const std::shared_ptr<arrow::Table>& table) const override {
    auto lvals = left->evaluate(table);
    auto rvals = right->evaluate(table);

    std::vector<std::shared_ptr<arrow::Scalar>> result;
    result.reserve(lvals.size());

    for (size_t i = 0; i < lvals.size(); i++) {
      // handle int operations
      auto l_int = std::dynamic_pointer_cast<arrow::Int64Scalar>(lvals[i]);
      auto r_int = std::dynamic_pointer_cast<arrow::Int64Scalar>(rvals[i]);

      if (l_int && r_int) {
        int64_t lv = l_int->value;
        int64_t rv = r_int->value;

        switch (op) {
          case OpType::ADD:
            result.push_back(std::make_shared<arrow::Int64Scalar>(lv + rv));
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
            throw std::runtime_error("Unsupported int operation");
        }
      }

      else {
        // boolean operations
        auto l_bool = std::dynamic_pointer_cast<arrow::BooleanScalar>(lvals[i]);
        auto r_bool = std::dynamic_pointer_cast<arrow::BooleanScalar>(rvals[i]);

        if (l_bool && r_bool) {
          bool lv = l_bool->value;
          bool rv = r_bool->value;

          switch (op) {
            case OpType::AND:
              result.push_back(
                  std::make_shared<arrow::BooleanScalar>(lv && rv));
              break;

            case OpType::OR:
              result.push_back(
                  std::make_shared<arrow::BooleanScalar>(lv || rv));
              break;

            default:
              throw std::runtime_error("Unsupported boolean operation");
          }
        } else {
          throw std::runtime_error("Type mismatch in BinaryExpr");
        }
      }
    }

    return result;
  }
};