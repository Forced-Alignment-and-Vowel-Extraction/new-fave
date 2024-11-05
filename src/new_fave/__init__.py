from new_fave.measurements.vowel_measurement import VowelMeasurement, \
    VowelClass, \
    VowelClassCollection, \
    SpeakerCollection

from new_fave.patterns.fave_audio_textgrid import fave_audio_textgrid
from new_fave.patterns.fave_corpus import fave_corpus
from new_fave.patterns.fave_subcorpora import fave_subcorpora
from new_fave.patterns.writers import write_data, pickle_speakers, unpickle_speakers

from importlib.metadata import version

# this is bugged for testing
#__version__ = version("new_fave")

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