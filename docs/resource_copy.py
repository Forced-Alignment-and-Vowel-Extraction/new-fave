from pathlib import Path

resource_path = Path("../src/new_fave/resources/")
all_resource = resource_path.glob("*")

for src in all_resource:
    dest = Path(src.parent.name, src.name)
    dest.write_text(src.read_text())
