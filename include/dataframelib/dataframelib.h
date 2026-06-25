#pragma once
// Compatibility macro for older Arrow test code
#ifndef ARROW_THROW_NOT_OK
#define ARROW_THROW_NOT_OK(expr)                             \
  do {                                                       \
    auto _st = (expr);                                       \
    if (!_st.ok()) throw std::runtime_error(_st.ToString()); \
  } while (0)
#endif
#include <arrow/api.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "dataframe.h"
#include "expr.h"
#include "lazy.h"

namespace dataframelib {

// ========================= I/O =========================
EagerDataFrame read_csv(const std::string& path);
EagerDataFrame read_parquet(const std::string& path);
LazyDataFrame scan_csv(const std::string& path);
LazyDataFrame scan_parquet(const std::string& path);

// ========================= FROM COLUMNS =========================
EagerDataFrame from_columns(
    const std::vector<std::pair<std::string, std::shared_ptr<arrow::Array>>>&
        cols);

}  // namespace dataframelib