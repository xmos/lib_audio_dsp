from audio_dsp.design.parse_json import DspJson
import json
from pathlib import Path
import os
from pydantic.json_schema import GenerateJsonSchema

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


class GenerateJsonSchema_noParams(GenerateJsonSchema):
    def generate(self, schema, mode='validation'):
        json_schema = super().generate(schema, mode=mode)
        to_pop = []
        for key, value in json_schema['$defs'].items():
            if "Parameters" in key or "biquad_" in key or "Fork" in key:
                # can't pop yet because we're iterating
                to_pop.append(key)
            elif "parameters" in value["properties"]:
                value["properties"].pop("parameters")
                json_schema['$defs'][key] = value

        for key in to_pop:
            json_schema['$defs'].pop(key)

        return json_schema

def test_no_params_schema():
    dsp_schema = DspJson.model_json_schema(schema_generator=GenerateJsonSchema_noParams)

    # write the JSON data to a file
    filepath = Path(schema_path, "test_no_params_schema.json")
    filepath.write_text(json.dumps(dsp_schema, indent=2))

    with open(Path(schema_path, "dsp_schema_no_params.json"), "r") as f:
        commited_schema = json.load(f)

    assert dsp_schema == commited_schema, "Generated schema does not match commited schema"

    os.remove(filepath)


if __name__ == "__main__":
    test_no_params_schema()