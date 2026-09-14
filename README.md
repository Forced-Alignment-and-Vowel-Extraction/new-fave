# new-fave

![PyPI](https://img.shields.io/pypi/v/new-fave.png) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/new-fave) [![Conda Version](https://img.shields.io/conda/vn/conda-forge/new-fave.svg)](https://anaconda.org/conda-forge/new-fave) [![European Galaxy server](https://img.shields.io/badge/usegalaxy-.eu-brightgreen?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABgAAAASCAYAAABB7B6eAAAABGdBTUEAALGPC/xhBQAAACBjSFJNAAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAACXBIWXMAAAsTAAALEwEAmpwYAAACC2lUWHRYTUw6Y29tLmFkb2JlLnhtcAAAAAAAPHg6eG1wbWV0YSB4bWxuczp4PSJhZG9iZTpuczptZXRhLyIgeDp4bXB0az0iWE1QIENvcmUgNS40LjAiPgogICA8cmRmOlJERiB4bWxuczpyZGY9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkvMDIvMjItcmRmLXN5bnRheC1ucyMiPgogICAgICA8cmRmOkRlc2NyaXB0aW9uIHJkZjphYm91dD0iIgogICAgICAgICAgICB4bWxuczp0aWZmPSJodHRwOi8vbnMuYWRvYmUuY29tL3RpZmYvMS4wLyI+CiAgICAgICAgIDx0aWZmOlJlc29sdXRpb25Vbml0PjI8L3RpZmY6UmVzb2x1dGlvblVuaXQ+CiAgICAgICAgIDx0aWZmOkNvbXByZXNzaW9uPjE8L3RpZmY6Q29tcHJlc3Npb24+CiAgICAgICAgIDx0aWZmOk9yaWVudGF0aW9uPjE8L3RpZmY6T3JpZW50YXRpb24+CiAgICAgICAgIDx0aWZmOlBob3RvbWV0cmljSW50ZXJwcmV0YXRpb24+MjwvdGlmZjpQaG90b21ldHJpY0ludGVycHJldGF0aW9uPgogICAgICA8L3JkZjpEZXNjcmlwdGlvbj4KICAgPC9yZGY6UkRGPgo8L3g6eG1wbWV0YT4KD0UqkwAAAn9JREFUOBGlVEuLE0EQruqZiftwDz4QYT1IYM8eFkHFw/4HYX+GB3/B4l/YP+CP8OBNTwpCwFMQXAQPKtnsg5nJZpKdni6/6kzHvAYDFtRUT71f3UwAEbkLch9ogQxcBwRKMfAnM1/CBwgrbxkgPAYqlBOy1jfovlaPsEiWPROZmqmZKKzOYCJb/AbdYLso9/9B6GppBRqCrjSYYaquZq20EUKAzVpjo1FzWRDVrNay6C/HDxT92wXrAVCH3ASqq5VqEtv1WZ13Mdwf8LFyyKECNbgHHAObWhScf4Wnj9CbQpPzWYU3UFoX3qkhlG8AY2BTQt5/EA7qaEPQsgGLWied0A8VKrHAsCC1eJ6EFoUd1v6GoPOaRAtDPViUr/wPzkIFV9AaAZGtYB568VyJfijV+ZBzlVZJ3W7XHB2RESGe4opXIGzRTdjcAupOK09RA6kzr1NTrTj7V1ugM4VgPGWEw+e39CxO6JUw5XhhKihmaDacU2GiR0Ohcc4cZ+Kq3AjlEnEeRSazLs6/9b/kh4eTC+hngE3QQD7Yyclxsrf3cpxsPXn+cFdenF9aqlBXMXaDiEyfyfawBz2RqC/O9WF1ysacOpytlUSoqNrtfbS642+4D4CS9V3xb4u8P/ACI4O810efRu6KsC0QnjHJGaq4IOGUjWTo/YDZDB3xSIxcGyNlWcTucb4T3in/3IaueNrZyX0lGOrWndstOr+w21UlVFokILjJLFhPukbVY8OmwNQ3nZgNJNmKDccusSb4UIe+gtkI+9/bSLJDjqn763f5CQ5TLApmICkqwR0QnUPKZFIUnoozWcQuRbC0Km02knj0tPYx63furGs3x/iPnz83zJDVNtdP3QAAAABJRU5ErkJggg==)](https://usegalaxy.eu/root?tool_id=new_fave_extract)[![Python CI](https://github.com/Forced-Alignment-and-Vowel-Extraction/new-fave/actions/workflows/test-and-run.yml/badge.svg)](https://github.com/Forced-Alignment-and-Vowel-Extraction/new-fave/actions/workflows/test-and-run.yml) [![codecov](https://codecov.io/gh/Forced-Alignment-and-Vowel-Extraction/new-fave/graph/badge.svg?token=8JRGOB9NMN)](https://codecov.io/gh/Forced-Alignment-and-Vowel-Extraction/new-fave) [![Maintainability](https://qlty.sh/badges/9a2ed726-373d-4c82-8740-2b27dea1bdd4/maintainability.svg)](https://qlty.sh/gh/Forced-Alignment-and-Vowel-Extraction/projects/new-fave) [![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) [![DOI](https://zenodo.org/badge/761302010.svg)](https://zenodo.org/badge/latestdoi/761302010)

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