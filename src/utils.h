#pragma once

#include <arrow/api.h>

#include <memory>
#include <string>
#include <unordered_map>

namespace dataframelib {

class EagerDataFrame;
class LazyDataFrame;

void sink_csv(const LazyDataFrame& df, const std::string& path);
void sink_parquet(const LazyDataFrame& df, const std::string& path);
}  // namespace dataframelib