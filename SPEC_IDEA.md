# Thunk Idears

100M param
2K context window
translator model
- English to English (semantic)
- code to code
- English to code
- code to English
- foreign-lang to English
- English to foreign-lang
- <lang> tag to indicate the output language

considerations
- multi-headed attention?
- turbo-quant?


## Usage
- direct translation
- encoder-decoder
  - novel text/code can be encoded and memorized in a DB
  - can use for semantic search _and_ creation


## Examples
- document -> document
  - foundational literature
  - LEAN proofs
  - high-value scripts
- English description of a CLI task -> bash script
- bash script -> powershell script
- document -> alternate, equivalent document
  - ESV Bible chapter -> NIV Bible chapter


## Restrictions
- primarily English
- top 100 books (Mortimer Adler's list)
  - multiple bible translations
  - masterworks
- languages
  - LEAN
  - Typescript
  - Python
  - bash, cmd, powershell
- top scientific papers
  - only the cream
  - papers with foreign language translations would be great
    - lang -> English, English -> lang
- top 100 codebases
  - curated for ubiquity and value
- distillation
  - common tasks: English -> script ... script -> English
