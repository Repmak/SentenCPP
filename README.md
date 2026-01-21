<p align="center"><img width="40%" src="docs/assets/sentencpp-logo.png" /></p>

## Overview
**sentenCPP** is a C++20 library designed to replicate the functionality and ease of use of the Python library `sentence-transformers`. It provides a complete pipeline from **text tokenization** to **vector embeddings**, extending to **mathematical operations** for analysis.

While NLP in C++ is entirely possible using various high-performance tools, the process of manually stitching these libraries together is often time-consuming and complex. **sentenCPP** is not intended to replace hyper-specialised libraries. It aims to eliminate the friction inherent to C++ development.


## Development Status
This project is still in development. As of right now, BERT, RoBERTa, and DistilBERT models are supported by the inference engine. Furthermore, the engine can perform semantic search, semantic analysis, and named entity extraction, but does not yet support autoregressive text generation or cross-encoding tasks. 


## Getting Started
This section will help you integrate **sentenCPP** into your first project. For more detailed information, please visit: [**sentenCPP - Documentation**](https://repmak.github.io/#/sentencpp-docs/).

#### 1. Prerequisites
Before installing **sentenCPP**, you need to set up two core dependencies:

- ICU (International Components for Unicode): Required for text normalisation. This can be installed via your package manager for macOS and Linux. For Windows, download the binary directly from the [**ICU Releases**](https://github.com/unicode-org/icu/releases).
- ONNX Runtime: This is the engine used to run the machine learning models. This can be download directly from the [**ONNX Runtime Releases**](https://github.com/microsoft/onnxruntime/releases). Extract it to a known directory.

#### 2. CMake Setup
The easiest way to include **sentenCPP** in your project is using CMake's `FetchContent` module. This will also automatically handle the `nlohmann_json` dependency.

```cmake
cmake_minimum_required(VERSION 3.20)
project(my_app LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)

include(FetchContent)
FetchContent_Declare(
        sentencpp
        GIT_REPOSITORY https://github.com/Repmak/sentenCPP.git
        GIT_TAG main
)
FetchContent_MakeAvailable(sentencpp)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE sentencpp)
```

#### 3. Configuring & Building
When building, you must provide the paths to your ICU and ONNX Runtime installations. Run the following commands from your project root:
```bash
mkdir build && cd build

cmake .. \
  -DICU_ROOT=/path/to/your/icu4c \
  -DONNXRUNTIME_ROOT=/path/to/your/onnxruntime-osx-arm64-1.23.2

cmake --build .
```

**Note:** If you have already configured `ICU_ROOT` and `ONNXRUNTIME_ROOT` as CMake options within your IDE, you do not need to pass them via the command line.

#### 4. Downloading the ONNX model
The Hugging Face `optimum` library can be used to export a model to the ONNX format. Install the dependencies and export the model using:
```bash
pip install "optimum[exporters]"
pip install "optimum[onnxruntime]"

optimum-cli export onnx --model sentence-transformers/all-MiniLM-L6-v2 --task default sentencpp_model/
```

#### 5. All Done!
The following snippet demonstrates how to tokenize text, generate embeddings, and calculate the cosine similarity between two sentences. Ensure the paths to `tokenizer.json` and `model.onnx` have been updated.

```cpp
#include <iostream>
#include <sentencpp/tokenizer/WordPiece.h>
#include <sentencpp/inference/OnnxEngine.h>
#include <sentencpp/embedding_utils/VectorMaths.h>

int main() {
    sentencpp::tokenizer::WordPieceConfig config;
    config.config_path = "/path/to/your/sentence-transformers-all-mini-lm-l6-v2/tokenizer.json";
    const sentencpp::tokenizer::WordPiece tokenizer(config);

    const std::string sentence_1 = "The cat sits outside";
    const std::string sentence_2 = "A feline is resting outdoors";

    const auto tokens_1 = tokenizer.tokenize(sentence_1);
    const auto tokens_2 = tokenizer.tokenize(sentence_2);

    sentencpp::inference::ModelConfig model_config;
    model_config.model_path = "/path/to/your/sentence-transformers-all-mini-lm-l6-v2/model.onnx";
    sentencpp::inference::OnnxEngine engine(model_config);

    std::vector<std::vector<float>> embeddings_1 = engine.encode(tokens_1);
    std::vector<std::vector<float>> embeddings_2 = engine.encode(tokens_2);

    auto vector_1 = sentencpp::embedding_utils::VectorMaths::mean_pooling(embeddings_1, tokens_1);
    auto vector_2 = sentencpp::embedding_utils::VectorMaths::mean_pooling(embeddings_2, tokens_2);

    float similarity = sentencpp::embedding_utils::VectorMaths::cosine_similarity(vector_1, vector_2);
    std::cout << "Similarity Score: " << similarity << std::endl;

    return 0;
}
```


## Suggestions & Feedback

Please feel free to open an issue or reach out!
