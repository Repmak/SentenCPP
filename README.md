# SentenCPP

Still in development!

## 1. Overview
SentenCPP is a C++20 library designed to replicate the ease of use of the Python library `sentence-transformers`. It provides an end-to-end pipeline from raw text tokenization to vector embeddings optimised for low-latency production environments where applications would otherwise be bottlenecked by Python's interpreter.


## 2. Get Started
This section will outline all the key details to get SentenCPP working on your machine.

### 2.1 Model Compatibility
SentenCPP supports any transformer model that uses WordPiece tokenization and follows the BERT/DistilBERT architecutre exported to ONNX. This includes models like `all-MiniLM-L6-v2` and `bert-base-uncased`. Exporting to ONNX can be done using the Hugging Face Optimum library. It provides a quick and easy way of exporting BERT models to the ONNX format.

**Step 1:** Install the following requirements.
```bash
pip install "optimum[exporters]"
pip install "optimum[onnxruntime]"
```

**Step 2:** Run the export. Substitute `all-MiniLM-L6-v2` for a model of your choice.
```bash
optimum-cli export onnx --model sentence-transformers/all-MiniLM-L6-v2 --task default sentencpp_model/
```
**Step 3:** Finally, your `sentencpp_model/` folder will contain the model (`model.onnx`), as well as various other files determining the model's configuration settings. These will be further addressed in section 3.

### 2.2 Configuring ICU4C
SentenCPP relies on `ICU4C` for text normalisation. You must ensure that CMake can locate the library on your system.

**Step 1:** Run the following build.
```bash
mkdir build && cd build
cmake .. -DICU_ROOT=/opt/homebrew/opt/icu4c
make
```

**Step 2:** Within CLion, go to: `Settings` > `Build, Execution, Deployment` > `CMake` and set `CMake Options` to `-DICU_ROOT=/opt/homebrew/opt/icu4c` (or the Win/Linux equivalent).


### 2.3 Other library dependencies
todo


## 3. Get Started
This section will outline all the key details to get SentenCPP working on your machine.


## 4. Get Started
This section will outline all the key details to get SentenCPP working on your machine.

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

Go to Settings > Build, Execution, Deployment > CMake and set CMake Options to '-DICU_ROOT=/opt/homebrew/opt/icu4c' (or the win/linus equivalent)


to export to onnx:
pip install "optimum[exporters]"
pip install optimum[onnxruntime]
optimum-cli export onnx --model dbmdz/bert-large-cased-finetuned-conll03-english --task token-classification bert_ner_onnx/
