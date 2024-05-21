# new-fave

![PyPI](https://img.shields.io/pypi/v/new-fave.png) [![Python CI](https://github.com/Forced-Alignment-and-Vowel-Extraction/new-fave/actions/workflows/test-and-run.yml/badge.svg)](https://github.com/Forced-Alignment-and-Vowel-Extraction/new-fave/actions/workflows/test-and-run.yml) [![codecov](https://codecov.io/gh/Forced-Alignment-and-Vowel-Extraction/new-fave/graph/badge.svg?token=8JRGOB9NMN)](https://codecov.io/gh/Forced-Alignment-and-Vowel-Extraction/new-fave) [![Maintainability](https://api.codeclimate.com/v1/badges/2f00920067765c0ad77f/maintainability)](https://codeclimate.com/github/Forced-Alignment-and-Vowel-Extraction/new-fave/maintainability) [![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## What is `new-fave`?

`new-fave` is a tool for automating and optimizing vowel formant extraction. It is philosophically similar (and named after) [the FAVE-suite](https://github.com/JoFrhwld/FAVE). However, `new-fave` has been completely written from scratch, and has some key differences from the FAVE-suite.

1. **`new-fave` does not include a forced-aligner.**
    It can process alignments produced by fave-align, 
    but we would recommend using the Monteal Forced Aligner instead
2. **`new-fave` does not require speaker demographics.**
    You can optionally pass `fave-extract` a speaker
    demographics file to be merged into your formant data,
    but this does *not* influence how the data is processed
    in any way. Besides including file name and speaker
    number data, you can pass *any* demographic information
    you would like.
3. **`new-fave` does not assume North American English vowels**.
    Your alignments can contain any set of vowels, in
    any transcription system, as long as you can provide 
    a regular expression to identify them.
4. **`new-fave` is customizable.**
    With config files, you can customize vowel recoding,
    labelset parsing, and point measurement heuristics.
5. **`new-fave` is focused on formant tracks.**
    You can still produce single point measurements 
    for vowels, but `new-fave` is built upon 
    the [FastTrack](https://fasttrackiverse.github.io/fasttrackpy/) method. By default, it will write 
    output files including point measurements, full
    formant tracks, and Discrete Cosine Transform 
    coefficients.
6. **`new-fave` is maintainable**. As time goes on, and the 
    code base needs updating, the organization and 
    infrastructure of `new-fave` should allow it to be
    readilly updateable.

You can read more at [the `new-fave` documentation](https://forced-alignment-and-vowel-extraction.github.io/new-fave/).

## Installation

You can install `new-fave` with `pip`.

```bash
# bash
pip install new-fave
```

## Usage

To use the default settings (which assume CMU 
dictionary transcriptions), you can use one of these 
patterns.

### A single audio + textgrid pair

```bash
# bash
fave-extract audio-textgrid speaker1.wav speaker1.TextGrid
```

### A directory of audio + textgrid pairs

```bash
# bash
fave-extract corpus speakers/
```

### Multiple subdirectories of audio + textgrid pairs

```bash
# bash
fave-extract subcorpora data/*
```