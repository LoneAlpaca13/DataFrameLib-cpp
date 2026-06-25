#pragma once
#include <arrow/api.h>

#include <memory>
#include <string>
#include <vector>

namespace dataframelib {

enum class OpType {
  ADD,
  SUB,
  MUL,
  DIV,
  MOD,
  EQ,
  NEQ,
  GT,
  LT,
  GTE,
  LTE,
  AND,
  OR,
  NOT,
  ABS,
  STR_CONTAINS,
  STR_STARTSWITH,
  STR_ENDSWITH,
  STR_UPPER,
  STR_LOWER,
  STR_LENGTH,
  IS_NULL,
  IS_NOT_NULL,
  AGG_SUM,
  AGG_MEAN,
  AGG_COUNT,
  AGG_MIN,
  AGG_MAX
};

class Expr {
 public:
  virtual ~Expr() = default;
  virtual std::vector<std::shared_ptr<arrow::Scalar>> evaluate(
      const std::shared_ptr<arrow::Table>& table) const = 0;
};

class LiteralExpr : public Expr {
 public:
  std::shared_ptr<arrow::Scalar> value;
  explicit LiteralExpr(std::shared_ptr<arrow::Scalar> v)
      : value(std::move(v)) {}
  std::vector<std::shared_ptr<arrow::Scalar>> evaluate(
      const std::shared_ptr<arrow::Table>& table) const override;
};

class ColumnExpr : public Expr {
 public:
  std::string name;
  explicit ColumnExpr(const std::string& n) : name(n) {}
  std::string getName() const { return name; }
  std::vector<std::shared_ptr<arrow::Scalar>> evaluate(
      const std::shared_ptr<arrow::Table>& table) const override;
};

class UnaryExpr : public Expr {
 public:
  std::shared_ptr<Expr> operand;
  OpType op;
  std::string str_arg;
  UnaryExpr(std::shared_ptr<Expr> o, OpType op_type, std::string arg = "")
      : operand(std::move(o)), op(op_type), str_arg(std::move(arg)) {}
  std::vector<std::shared_ptr<arrow::Scalar>> evaluate(
      const std::shared_ptr<arrow::Table>& table) const override;
};

class BinaryExpr : public Expr {
 private:
  std::shared_ptr<Expr> left_;
  std::shared_ptr<Expr> right_;
  OpType op_;

 public:
  BinaryExpr(std::shared_ptr<Expr> l, std::shared_ptr<Expr> r, OpType o)
      : left_(std::move(l)), right_(std::move(r)), op_(o) {}
  std::shared_ptr<Expr> getLeft() const { return left_; }
  std::shared_ptr<Expr> getRight() const { return right_; }
  OpType getOp() const { return op_; }
  std::vector<std::shared_ptr<arrow::Scalar>> evaluate(
      const std::shared_ptr<arrow::Table>& table) const override;
};

class AggExpr : public Expr {
 public:
  std::shared_ptr<Expr> input;
  OpType agg_op;
  AggExpr(std::shared_ptr<Expr> in, OpType op)
      : input(std::move(in)), agg_op(op) {}
  std::vector<std::shared_ptr<arrow::Scalar>> evaluate(
      const std::shared_ptr<arrow::Table>& table) const override;
};

// ========================= EXPRPTR WRAPPER =========================
// Wraps shared_ptr<Expr> and adds method chaining (.abs(), .contains(), etc.)
struct ExprPtr {
  std::shared_ptr<Expr> ptr;

  ExprPtr(std::shared_ptr<Expr> p) : ptr(std::move(p)) {}
  operator std::shared_ptr<Expr>() const { return ptr; }

  ExprPtr abs() const {
    return ExprPtr(std::make_shared<UnaryExpr>(ptr, OpType::ABS));
  }
  ExprPtr is_null() const {
    return ExprPtr(std::make_shared<UnaryExpr>(ptr, OpType::IS_NULL));
  }
  ExprPtr is_not_null() const {
    return ExprPtr(std::make_shared<UnaryExpr>(ptr, OpType::IS_NOT_NULL));
  }
  ExprPtr contains(const std::string& s) const {
    return ExprPtr(std::make_shared<UnaryExpr>(ptr, OpType::STR_CONTAINS, s));
  }
  ExprPtr startswith(const std::string& s) const {
    return ExprPtr(std::make_shared<UnaryExpr>(ptr, OpType::STR_STARTSWITH, s));
  }
  ExprPtr endswith(const std::string& s) const {
    return ExprPtr(std::make_shared<UnaryExpr>(ptr, OpType::STR_ENDSWITH, s));
  }
  ExprPtr upper() const {
    return ExprPtr(std::make_shared<UnaryExpr>(ptr, OpType::STR_UPPER));
  }
  ExprPtr lower() const {
    return ExprPtr(std::make_shared<UnaryExpr>(ptr, OpType::STR_LOWER));
  }
  ExprPtr length() const {
    return ExprPtr(std::make_shared<UnaryExpr>(ptr, OpType::STR_LENGTH));
  }
  ExprPtr sum() const {
    return ExprPtr(std::make_shared<AggExpr>(ptr, OpType::AGG_SUM));
  }
  ExprPtr mean() const {
    return ExprPtr(std::make_shared<AggExpr>(ptr, OpType::AGG_MEAN));
  }
  ExprPtr count() const {
    return ExprPtr(std::make_shared<AggExpr>(ptr, OpType::AGG_COUNT));
  }
  ExprPtr min() const {
    return ExprPtr(std::make_shared<AggExpr>(ptr, OpType::AGG_MIN));
  }
  ExprPtr max() const {
    return ExprPtr(std::make_shared<AggExpr>(ptr, OpType::AGG_MAX));
  }
  ExprPtr alias(const std::string&) const { return *this; }
  // Aliases for TA test programs
  ExprPtr to_upper() const { return upper(); }
  ExprPtr to_lower() const { return lower(); }
  ExprPtr starts_with(const std::string& s) const { return startswith(s); }
  ExprPtr ends_with(const std::string& s) const { return endswith(s); }
};

// ========================= FACTORY FUNCTIONS =========================
inline ExprPtr col(const std::string& name) {
  return ExprPtr(std::make_shared<ColumnExpr>(name));
}
inline ExprPtr lit(double v) {
  return ExprPtr(std::make_shared<LiteralExpr>(arrow::MakeScalar(v)));
}
inline ExprPtr lit(int v) {
  return ExprPtr(std::make_shared<LiteralExpr>(arrow::MakeScalar((int64_t)v)));
}
inline ExprPtr lit(int64_t v) {
  return ExprPtr(std::make_shared<LiteralExpr>(arrow::MakeScalar(v)));
}
inline ExprPtr lit(const std::string& v) {
  return ExprPtr(std::make_shared<LiteralExpr>(arrow::MakeScalar(v)));
}
inline ExprPtr lit(const char* v) { return lit(std::string(v)); }

// ========================= ARITHMETIC =========================
inline ExprPtr operator+(ExprPtr a, ExprPtr b) {
  return ExprPtr(std::make_shared<BinaryExpr>(a.ptr, b.ptr, OpType::ADD));
}
inline ExprPtr operator-(ExprPtr a, ExprPtr b) {
  return ExprPtr(std::make_shared<BinaryExpr>(a.ptr, b.ptr, OpType::SUB));
}
inline ExprPtr operator*(ExprPtr a, ExprPtr b) {
  return ExprPtr(std::make_shared<BinaryExpr>(a.ptr, b.ptr, OpType::MUL));
}
inline ExprPtr operator/(ExprPtr a, ExprPtr b) {
  return ExprPtr(std::make_shared<BinaryExpr>(a.ptr, b.ptr, OpType::DIV));
}
inline ExprPtr operator%(ExprPtr a, ExprPtr b) {
  return ExprPtr(std::make_shared<BinaryExpr>(a.ptr, b.ptr, OpType::MOD));
}
inline ExprPtr operator+(ExprPtr a, double b) { return a + lit(b); }
inline ExprPtr operator+(double a, ExprPtr b) { return lit(a) + b; }
inline ExprPtr operator-(ExprPtr a, double b) { return a - lit(b); }
inline ExprPtr operator-(double a, ExprPtr b) { return lit(a) - b; }
inline ExprPtr operator*(ExprPtr a, double b) { return a * lit(b); }
inline ExprPtr operator*(double a, ExprPtr b) { return lit(a) * b; }
inline ExprPtr operator/(ExprPtr a, double b) { return a / lit(b); }
inline ExprPtr operator/(double a, ExprPtr b) { return lit(a) / b; }
inline ExprPtr operator+(ExprPtr a, int b) { return a + lit(b); }
inline ExprPtr operator+(int a, ExprPtr b) { return lit(a) + b; }
inline ExprPtr operator-(ExprPtr a, int b) { return a - lit(b); }
inline ExprPtr operator-(int a, ExprPtr b) { return lit(a) - b; }
inline ExprPtr operator*(ExprPtr a, int b) { return a * lit(b); }
inline ExprPtr operator*(int a, ExprPtr b) { return lit(a) * b; }
inline ExprPtr operator/(ExprPtr a, int b) { return a / lit(b); }
inline ExprPtr operator%(ExprPtr a, int b) { return a % lit(b); }

// ========================= COMPARISON =========================
inline ExprPtr operator==(ExprPtr a, ExprPtr b) {
  return ExprPtr(std::make_shared<BinaryExpr>(a.ptr, b.ptr, OpType::EQ));
}
inline ExprPtr operator!=(ExprPtr a, ExprPtr b) {
  return ExprPtr(std::make_shared<BinaryExpr>(a.ptr, b.ptr, OpType::NEQ));
}
inline ExprPtr operator>(ExprPtr a, ExprPtr b) {
  return ExprPtr(std::make_shared<BinaryExpr>(a.ptr, b.ptr, OpType::GT));
}
inline ExprPtr operator<(ExprPtr a, ExprPtr b) {
  return ExprPtr(std::make_shared<BinaryExpr>(a.ptr, b.ptr, OpType::LT));
}
inline ExprPtr operator>=(ExprPtr a, ExprPtr b) {
  return ExprPtr(std::make_shared<BinaryExpr>(a.ptr, b.ptr, OpType::GTE));
}
inline ExprPtr operator<=(ExprPtr a, ExprPtr b) {
  return ExprPtr(std::make_shared<BinaryExpr>(a.ptr, b.ptr, OpType::LTE));
}
inline ExprPtr operator==(ExprPtr a, double b) { return a == lit(b); }
inline ExprPtr operator!=(ExprPtr a, double b) { return a != lit(b); }
inline ExprPtr operator>(ExprPtr a, double b) { return a > lit(b); }
inline ExprPtr operator<(ExprPtr a, double b) { return a < lit(b); }
inline ExprPtr operator>=(ExprPtr a, double b) { return a >= lit(b); }
inline ExprPtr operator<=(ExprPtr a, double b) { return a <= lit(b); }
inline ExprPtr operator==(ExprPtr a, int b) { return a == lit(b); }
inline ExprPtr operator!=(ExprPtr a, int b) { return a != lit(b); }
inline ExprPtr operator>(ExprPtr a, int b) { return a > lit(b); }
inline ExprPtr operator<(ExprPtr a, int b) { return a < lit(b); }
inline ExprPtr operator>=(ExprPtr a, int b) { return a >= lit(b); }
inline ExprPtr operator<=(ExprPtr a, int b) { return a <= lit(b); }
inline ExprPtr operator==(ExprPtr a, const std::string& b) {
  return a == lit(b);
}
inline ExprPtr operator!=(ExprPtr a, const std::string& b) {
  return a != lit(b);
}
inline ExprPtr operator==(ExprPtr a, const char* b) {
  return a == std::string(b);
}
inline ExprPtr operator!=(ExprPtr a, const char* b) {
  return a != std::string(b);
}

// ========================= BOOLEAN =========================
inline ExprPtr operator&(ExprPtr a, ExprPtr b) {
  return ExprPtr(std::make_shared<BinaryExpr>(a.ptr, b.ptr, OpType::AND));
}
inline ExprPtr operator|(ExprPtr a, ExprPtr b) {
  return ExprPtr(std::make_shared<BinaryExpr>(a.ptr, b.ptr, OpType::OR));
}
inline ExprPtr operator~(ExprPtr a) {
  return ExprPtr(std::make_shared<UnaryExpr>(a.ptr, OpType::NOT));
}

// ========================= FREE FUNCTION ALIASES =========================
inline ExprPtr abs_expr(ExprPtr e) { return e.abs(); }
inline ExprPtr is_null(ExprPtr e) { return e.is_null(); }
inline ExprPtr is_not_null(ExprPtr e) { return e.is_not_null(); }
inline ExprPtr str_contains(ExprPtr e, const std::string& s) {
  return e.contains(s);
}
inline ExprPtr str_startswith(ExprPtr e, const std::string& s) {
  return e.startswith(s);
}
inline ExprPtr str_endswith(ExprPtr e, const std::string& s) {
  return e.endswith(s);
}
inline ExprPtr str_upper(ExprPtr e) { return e.upper(); }
inline ExprPtr str_lower(ExprPtr e) { return e.lower(); }
inline ExprPtr str_length(ExprPtr e) { return e.length(); }
inline ExprPtr agg_sum(ExprPtr e) { return e.sum(); }
inline ExprPtr agg_mean(ExprPtr e) { return e.mean(); }
inline ExprPtr agg_count(ExprPtr e) { return e.count(); }
inline ExprPtr agg_min(ExprPtr e) { return e.min(); }
inline ExprPtr agg_max(ExprPtr e) { return e.max(); }

}  // namespace dataframelib