from pathlib import Path
import toml
import new_fave

def test_version():
    pyproject_toml_file = Path(__file__).parent.parent / "pyproject.toml"
    toml_v = "unknown"
    if pyproject_toml_file.exists() and pyproject_toml_file.is_file():
        data = toml.load(pyproject_toml_file)
        # check project.version
        if "project" in data and "version" in data["project"]:
            toml_v = data["project"]["version"]
        # check tool.poetry.version
        elif "tool" in data and "poetry" in data["tool"] and "version" in data["tool"]["poetry"]:
            toml_v = data["tool"]["poetry"]["version"]
    
    assert new_fave.__version__ == toml_v
    