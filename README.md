# SentenCPP

Still in development!

## Overview

SentenCPP is a C++20 library designed to replicate the ease of use of Python's `sentence-transformers`. It provides an end-to-end pipeline from raw text tokenization to vector embeddings optimised for low-latency production environments where applications would otherwise be bottlenecked by Python's interpreter.


## Get Started


##



## Suggestions & Feedback

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
