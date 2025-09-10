from audio_dsp.design.parse_json import DspJson
import json
from pathlib import Path
import os

schema_path = Path(__file__).parent / "schemas"

def test_json_schema():
    dsp_schema = DspJson.model_json_schema()

    # write the JSON data to a file
    filepath = Path(schema_path, "test_dsp_schema.json")
    filepath.write_text(json.dumps(dsp_schema, indent=2))

    with open(Path(schema_path, "dsp_schema.json"), "r") as f:
        commited_schema = json.load(f)

    assert dsp_schema == commited_schema, "Generated schema does not match commited schema"

    os.remove(filepath)

if __name__ == "__main__":
    test_json_schema()