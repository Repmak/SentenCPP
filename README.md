<p align="center"><img width="40%" src="docs/assets/sentencpp-logo.png" /></p>

Still in development!

todo:
- write up api reference in a separate webpage for better clarity (it is a mess below!)
- set token segment ids to the correct value depending on use case (segment ids are always set to 0 right now)
- handle sequences which exceed max token length (use overlap and pass each batch into the onnx model)
- use prefix trie to avoid o(n^2) of max match algo

todo for much later?
- implement bpe and unigram tokenizers (for compatibility with other model architectures)
- support bin/h5 files?

## 1. Overview
sentenCPP is a C++20 library designed to replicate the ease of use of the Python library sentence-transformers. It provides a complete pipeline from text tokenization to vector embeddings, extending to mathematical operations for analysis.

<br/>

## 2. Getting Started
This section will outline all the key details to get sentenCPP working on your machine.


### 2.1 Model Compatibility
sentenCPP supports any transformer model that uses WordPiece tokenization and follows the BERT/DistilBERT architecture exported to ONNX. This includes models like `all-MiniLM-L6-v2` and `bert-base-uncased`. Exporting to ONNX can be done using Hugging Face Optimum. It provides a quick and easy way of exporting models to the ONNX format.

**Step 1**: Install the following requirements.
```bash
pip install "optimum[exporters]"
pip install "optimum[onnxruntime]"
```

**Step 2**: Run the export, substituting `all-MiniLM-L6-v2` for a model of your choice.
```bash
optimum-cli export onnx --model sentence-transformers/all-MiniLM-L6-v2 --task default sentencpp_model/
```

**Step 3**: Finally, your `sentencpp_model/` folder will contain the model (`model.onnx`), as well as various other files determining the model's configuration settings.

### 2.2 Configuring ICU4C
sentenCPP relies on ICU4C for text normalisation. If you already have ICU4C in your system's default library path you may be able to skip the steps below.

**Step 1**: Run the following build, substituting `PATH_TO_ICU` with the appropriate `ICU_ROOT` for your operating system. For example, on macOS (Homebrew), this will be `/opt/homebrew/opt/icu4c`.
```bash
mkdir build && cd build
cmake .. -DICU_ROOT=<PATH_TO_ICU>  # Substitute!
cmake --build .
```

**Step 2**: Within CLion, navigate to `Settings` > `Build, Execution, Deployment` > `CMake`.

**Step 3**: Set the `CMake Options` field to `-DICU_ROOT=/opt/homebrew/opt/icu4c`.

### 2.3 Other library dependencies
todo

<br/>

## 3. Example Usage
todo

<br/>

## 4. API Reference
[**API Reference**](https://repmak.github.io/#/sentencpp-docs/)

## 5. Suggestions & Feedback

Please feel free to open an issue or reach out!
