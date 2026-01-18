<p align="center"><img width="50%" src="docs/assets/sentencpp-logo.png" /></p>

Still in development! (sorry the content below is also not done yet!)

<br/>

## 1. Overview
sentenCPP is a C++20 library designed to replicate the ease of use of the Python library `sentence-transformers`. It provides an end-to-end pipeline from raw text tokenization to vector embeddings optimised for low-latency production environments where applications would otherwise be bottlenecked by Python's interpreter.

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

**Step 3**: Finally, your `sentencpp_model/` folder will contain the model (`model.onnx`), as well as various other files determining the model's configuration settings. These will be further addressed in the [**API Reference**](#4-api-reference).

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
sentenCPP is organised into three primary namespaces to handle the following distinct stages: 
- [**4.1 Tokenizer**](#41-sentencpptokenizer)
- [**4.2 Inference**](#42-sentencppinference)
- [**4.3 Embedding Utils**](#43-sentencppembedding_utils)

### 4.1 `sentencpp::tokenizer`
This handles the conversion of raw strings into sequences of tokens compatible with transformer models.

#### 4.1.1 `class WordPiece`
This is a subclass of `TokenizerInterface`. This class performs WordPiece tokenization.

**`WordPiece(const WordPieceConfig& config)`**: Initialises the tokenizer using [4.1.2 WordPieceConfig](#412-struct-wordpiececonfig).

**`std::vector<Token> tokenize(const std::string& text)`**: Normalises the string, converts it into a vector of tokens, and post-processes it.

#### 4.1.2 `struct WordPieceConfig`
This is a sub-struct of `TokenizerBaseConfig`. The members of this struct determine the behaviour and operations of the WordPiece tokenizer.

**`std::size_t max_input_chars_per_word = 100`**: This value should match your selected model's configuration settings. Sets a limit on the number of characters of a word. A word with length exceeding this limit will automatically be represented as an `UNK` token.

**`std::size_t max_length = 128`**: This value should match your selected model's configuration settings. Sets a limit on the number of tokens for a sequence. Tokens beyond this limit will be truncated. Note that 2 indices are reserved for special tokens; index 0 stores a `CLS` token, adn index 127 stores a `SEP` token.

**`bool to_lowercase = true`**: This value should match your selected model's configuration settings.

**`bool strip_accents = true`**: This value should match your selected model's configuration settings.

**`bool clean_text = true`**: This value should match your selected model's configuration settings.

**`bool handle_chinese_chars = true`**: This value should match your selected model's configuration settings.

**`std::string padding_token = "[PAD]"`**: This value should match your selected model's special token vocabulary.

**`std::string unknown_token = "[UNK]"`**: This value should match your selected model's special token vocabulary.

**`std::string classification_token = "[CLS]"`**: This value should match your selected model's special token vocabulary.

**`std::string separator_token = "[SEP]"`**: This value should match your selected model's special token vocabulary.

**`std::string mask_token = "[MASK]"`**: This value should match your selected model's special token vocabulary.

### 4.2 `sentencpp::inference`
This namespace performs the execution of ONNX models through ONNX Runtime.


### 4.3 `sentencpp::embedding_utils`
Contains static methods for mathematical operations on vectors.


<br/>

## 5. Suggestions & Feedback

Please feel free to open an issue or reach out!


todo:
- fix token segment ids
- handle sequences which exceed max token length (use overlap and pass each batch into the onnx model)
- use prefix trie to avoid o(n^2) of max match algo
- implement bpe and unigram tokenizers (not necessary for bert though)
- support bin/h5 files?

done:
- vocab list class for storing model's vocabulary
- max match tokenizer (and most of the normalisation required)
- onnx engine (basically ort wrapper) to get embeddings for a sequence of tokens
