# DataFrameLib — COP290 Assignment 4

A high-performance C++ DataFrame library backed by Apache Arrow, supporting both eager and lazy execution with a query optimiser.

---

## Project Structure

```
project/
├── CMakeLists.txt
├── include/
│   └── dataframelib/
│       ├── dataframe.h       # EagerDataFrame, GroupedDataFrame
│       ├── dataframelib.h    # Top-level include (I/O + all headers)
│       ├── expr.h            # Expression system (col, lit, operators)
│       └── lazy.h            # LazyDataFrame + Operation types
├── src/
│   ├── dataframe.cpp         # EagerDataFrame implementation
│   ├── expr.cpp              # Expression evaluation
│   ├── lazy.cpp              # LazyDataFrame + query optimiser
│   └── utils.cpp             # I/O helpers (read_csv, read_parquet, …)
├── README.md
└── report.pdf
```

---

## Dependencies

| Dependency | Purpose |
|---|---|
| **Apache Arrow** (≥ 12.0) | Columnar in-memory storage, CSV/Parquet I/O |
| **Parquet** (bundled with Arrow) | Parquet file support |
| **Graphviz** (optional) | Render `.dot` DAGs to PNG via `explain()` |
| **CMake** ≥ 3.16 | Build system |
| **C++20** compiler | `g++` ≥ 10 or `clang++` ≥ 11 |

### Installing Arrow (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install -y libarrow-dev libparquet-dev
```

### Installing Graphviz (optional, for DAG rendering)

```bash
sudo apt install -y graphviz
```

---

## Building

The library is built and tested via the TA-provided autograder. From the autograder directory:

```bash
cd path/to/A4-Tester
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 autograder.py --student-dir "/path/to/project"
```

To build the library on its own (without the autograder):

```bash
cd project
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel
```

If Arrow is installed in a non-standard location, pass its prefix:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/path/to/arrow
```

---

## API Overview

### I/O

```cpp
#include "dataframelib/dataframelib.h"
using namespace dataframelib;

EagerDataFrame df  = read_csv("data.csv");
EagerDataFrame df2 = read_parquet("data.parquet");
df.write_csv("out.csv");
df.write_parquet("out.parquet");

LazyDataFrame ldf = scan_csv("data.csv");
ldf.sink_csv("out.csv");
ldf.sink_parquet("out.parquet");

EagerDataFrame df3 = from_columns({{"col_name", arr}});
```

### Eager Operations

```cpp
df.select({"name", "salary"})
  .filter(col("age") > 30)
  .with_column("bonus", col("salary") * 0.1)
  .sort({"salary"}, false)
  .head(10);

df.group_by({"department"})
  .aggregate({{"salary_sum", "sum"}, {"age_mean", "mean"}});

df.join(other_df, {"id"}, "inner");   // "inner" | "left" | "outer"
```

### Lazy Operations

```cpp
auto result = scan_parquet("big.parquet")
    .filter(col("age") > 30)
    .select({"name", "salary"})
    .group_by({"dept"})
    .aggregate({{"avg_sal", "mean"}})
    .sort({"avg_sal"}, false)
    .head(5);

result.explain("plan.dot");
EagerDataFrame out = result.collect();
```

### Expression System

```cpp
col("salary") * 0.1
col("a") + col("b") - lit(100)
col("id") % 10

(col("age") > 30) & (col("dept") == "Engineering")
~(col("salary") < 50000.0)
col("x").is_null()

col("email").contains("@gmail.com")
col("name").upper()
col("name").lower()
col("name").length()

col("salary").sum()
col("salary").mean()
col("id").count()
```

---

## Submission

Package the project as follows before submitting on Moodle:

```bash
# Place all files inside a directory named 'project'
# Then from the parent of that directory:
tar -cvf <entry_number>.tar project/
```

The resulting archive should have this layout:

```
<entry_number>.tar
└── project/
    ├── CMakeLists.txt
    ├── include/
    │   └── dataframelib/
    ├── src/
    ├── README.md
    └── report.pdf
```