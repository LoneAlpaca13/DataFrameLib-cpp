#pragma once
#include <arrow/api.h>

std::shared_ptr<arrow::Table> readCSV(const std::string& filename);