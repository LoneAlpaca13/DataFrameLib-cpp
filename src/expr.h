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