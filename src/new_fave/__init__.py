from new_fave.measurements.vowel_measurement import VowelMeasurement, \
    VowelClass, \
    VowelClassCollection, \
    SpeakerCollection

from new_fave.patterns.fave_audio_textgrid import fave_audio_textgrid
from new_fave.patterns.fave_corpus import fave_corpus
from new_fave.patterns.fave_subcorpora import fave_subcorpora
from new_fave.patterns.writers import write_data, pickle_speakers, unpickle_speakers

from importlib.metadata import version

from pathlib import Path
import toml

__version__ = "unknown"
# adopt path to your pyproject.toml
pyproject_toml_file = Path(__file__).parent.parent.parent / "pyproject.toml"
if pyproject_toml_file.exists() and pyproject_toml_file.is_file():
    data = toml.load(pyproject_toml_file)
    # check project.version
    if "tool" in data and "poetry" in data["tool"] and "version" in data["tool"]["poetry"]:
        __version__ = data["tool"]["poetry"]["version"]

__all__ = [
    "VowelMeasurement", 
    "VowelClass", 
    "VowelClassCollection", 
    "SpeakerCollection",
    "fave_audio_textgrid",
    "fave_corpus",
    "fave_subcorpora",
    "write_data",
    "pickle_speakers",
    "unpickle_speakers"
]