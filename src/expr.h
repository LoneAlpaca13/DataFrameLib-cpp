#pragma once
#include <memory>
#include <string>

namespace dataframelib {

// ================= ENUM =================

enum class OpType { ADD, SUB, MUL, DIV, EQ, NEQ, GT, LT, GTE, LTE, AND, OR };

// ================= BASE =================

class Expr {
 public:
  virtual ~Expr() = default;
};

// ================= LITERAL =================

class LiteralExpr : public Expr {
 public:
  double num_value = 0.0;
  std::string str_value;
  bool is_string = false;

  LiteralExpr(double v) : num_value(v), is_string(false) {}
  LiteralExpr(const std::string& s) : str_value(s), is_string(true) {}
};

// ================= COLUMN =================

class ColumnExpr : public Expr {
 public:
  std::string name;

  ColumnExpr(const std::string& n) : name(n) {}
};

// ================= BINARY =================

class BinaryExpr : public Expr {
 private:
  std::shared_ptr<Expr> left;
  std::shared_ptr<Expr> right;
  OpType op;

 public:
  BinaryExpr(std::shared_ptr<Expr> l, std::shared_ptr<Expr> r, OpType o)
      : left(std::move(l)), right(std::move(r)), op(o) {}

  std::shared_ptr<Expr> getLeft() const { return left; }
  std::shared_ptr<Expr> getRight() const { return right; }
  OpType getOp() const { return op; }
};

// ================= HELPERS =================

inline std::shared_ptr<Expr> col(const std::string& name) {
  return std::make_shared<ColumnExpr>(name);
}

inline std::shared_ptr<Expr> lit(double v) {
  return std::make_shared<LiteralExpr>(v);
}

inline std::shared_ptr<Expr> lit(int v) {
  return std::make_shared<LiteralExpr>((double)v);
}

inline std::shared_ptr<Expr> lit(const std::string& v) {
  return std::make_shared<LiteralExpr>(v);
}

inline std::shared_ptr<Expr> lit(const char* v) { return lit(std::string(v)); }

// ================= SAFE OPERATORS =================

// Expr ↔ Expr
inline std::shared_ptr<Expr> operator==(std::shared_ptr<Expr> a,
                                        std::shared_ptr<Expr> b) {
  return std::make_shared<BinaryExpr>(a, b, OpType::EQ);
}

inline std::shared_ptr<Expr> operator!=(std::shared_ptr<Expr> a,
                                        std::shared_ptr<Expr> b) {
  return std::make_shared<BinaryExpr>(a, b, OpType::NEQ);
}

// Expr ↔ string
inline std::shared_ptr<Expr> operator==(std::shared_ptr<Expr> a,
                                        const std::string& b) {
  return std::make_shared<BinaryExpr>(a, lit(b), OpType::EQ);
}

inline std::shared_ptr<Expr> operator!=(std::shared_ptr<Expr> a,
                                        const std::string& b) {
  return std::make_shared<BinaryExpr>(a, lit(b), OpType::NEQ);
}

// Expr ↔ const char*
inline std::shared_ptr<Expr> operator==(std::shared_ptr<Expr> a,
                                        const char* b) {
  return a == std::string(b);
}

inline std::shared_ptr<Expr> operator!=(std::shared_ptr<Expr> a,
                                        const char* b) {
  return a != std::string(b);
}

// string ↔ Expr (reverse)
inline std::shared_ptr<Expr> operator==(const std::string& a,
                                        std::shared_ptr<Expr> b) {
  return std::make_shared<BinaryExpr>(lit(a), b, OpType::EQ);
}

inline std::shared_ptr<Expr> operator!=(const std::string& a,
                                        std::shared_ptr<Expr> b) {
  return std::make_shared<BinaryExpr>(lit(a), b, OpType::NEQ);
}

inline std::shared_ptr<Expr> operator==(const char* a,
                                        std::shared_ptr<Expr> b) {
  return std::string(a) == b;
}

inline std::shared_ptr<Expr> operator!=(const char* a,
                                        std::shared_ptr<Expr> b) {
  return std::string(a) != b;
}

// Expr ↔ number
inline std::shared_ptr<Expr> operator==(std::shared_ptr<Expr> a, double b) {
  return std::make_shared<BinaryExpr>(a, lit(b), OpType::EQ);
}

inline std::shared_ptr<Expr> operator!=(std::shared_ptr<Expr> a, double b) {
  return std::make_shared<BinaryExpr>(a, lit(b), OpType::NEQ);
}

inline std::shared_ptr<Expr> operator==(std::shared_ptr<Expr> a, int b) {
  return a == (double)b;
}

inline std::shared_ptr<Expr> operator!=(std::shared_ptr<Expr> a, int b) {
  return a != (double)b;
}

}  // namespace dataframelib