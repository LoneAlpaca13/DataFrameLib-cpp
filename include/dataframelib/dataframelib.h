#pragma once

#include <arrow/api.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// Core classes
#include "dataframe.h"
#include "expr.h"
#include "lazy.h"

namespace dataframelib {

// =====================
// I/O FUNCTIONS
// =====================

EagerDataFrame read_csv(const std::string& path);
EagerDataFrame read_parquet(const std::string& path);

LazyDataFrame scan_csv(const std::string& path);
LazyDataFrame scan_parquet(const std::string& path);

// =====================
// OUTPUT (EAGER + LAZY)
// =====================

// Eager
// (methods inside class, just listed here for clarity)

// Lazy sink
void sink_csv(const LazyDataFrame& df, const std::string& path);
void sink_parquet(const LazyDataFrame& df, const std::string& path);

// =====================
// BUILD FROM COLUMNS
// =====================

EagerDataFrame from_columns(
    const std::unordered_map<std::string, std::shared_ptr<arrow::Array>>& cols);

}  // namespace dataframelib