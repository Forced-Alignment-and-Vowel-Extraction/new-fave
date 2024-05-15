from importlib.resources import files

fave_fasttrack = str(files("new_fave").joinpath("resources", "fasttrack_config.yml"))

fave_cmu2phila = str(files("new_fave").joinpath("resources", "cmu2phila.yml"))
fave_cmu2labov = str(files("new_fave").joinpath("resources", "cmu2labov.yml"))

cmu_parser = str(files("new_fave").joinpath("resources", "cmu_parser.yml"))

fave_measurement = str(files("new_fave").joinpath("resources", "fave_measurement.yml"))

recodes = {
    "cmu2phila": fave_cmu2phila,
    "cmu2labov": fave_cmu2labov
}

parsers = {
    "cmu_parser": cmu_parser
}

heuristics = {
    "fave": fave_measurement
}

fasttrack_config = {
    "default": fave_fasttrack
}