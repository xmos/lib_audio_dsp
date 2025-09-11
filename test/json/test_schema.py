# Copyright 2025 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""Generate the JSON schema."""

from audio_dsp.design.parse_json import DspJson
import json
from pathlib import Path
import os
from pydantic.json_schema import GenerateJsonSchema

schema_path = Path(__file__).parent / "schemas"



class GenerateJsonSchema_simplified(GenerateJsonSchema):
    # strip out title, type, and default when there is const
    def generate(self, schema, mode='validation'):
        json_schema = super().generate(schema, mode=mode)

        for key, value in json_schema['$defs'].items():
            if "title" in value:
                value.pop("title")
            if "type" in value:
                value.pop("type")
            for this_property in value["properties"].values():
                if "title" in this_property:
                    this_property.pop("title")
                if "type" in this_property:
                    this_property.pop("type")
                if "default" in this_property and "const" in this_property:
                    this_property.pop("default")

            json_schema['$defs'][key] = value

        return json_schema

def test_json_schema():
    dsp_schema = DspJson.model_json_schema(schema_generator=GenerateJsonSchema_simplified)

    # write the JSON data to a file
    filepath = Path(schema_path, "test_dsp_schema.json")
    filepath.write_text(json.dumps(dsp_schema, indent=2))

    with open(Path(schema_path, "dsp_schema.json"), "r") as f:
        committed_schema = json.load(f)

    assert dsp_schema == committed_schema, "Generated schema does not match committed schema"

    os.remove(filepath)


class GenerateJsonSchema_noParams(GenerateJsonSchema_simplified):
    def generate(self, schema, mode='validation'):
        json_schema = super().generate(schema, mode=mode)
        to_pop = []
        for key, value in json_schema['$defs'].items():
            if "Parameters" in key or "biquad_" in key or "Fork" in key:
                # can't pop yet because we're iterating
                to_pop.append(key)
                continue
            if "parameters" in value["properties"]:
                value["properties"].pop("parameters")
                json_schema['$defs'][key] = value

        for key in to_pop:
            json_schema['$defs'].pop(key)

        return json_schema

def test_no_params_schema():
    dsp_schema = DspJson.model_json_schema(schema_generator=GenerateJsonSchema_noParams)

    # write the JSON data to a file
    filepath = Path(schema_path, "test_dsp_schema_no_params.json")
    filepath.write_text(json.dumps(dsp_schema, indent=2))

    with open(Path(schema_path, "dsp_schema_no_params.json"), "r") as f:
        committed_schema = json.load(f)

    assert dsp_schema == committed_schema, "Generated schema does not match committed schema"

    os.remove(filepath)


if __name__ == "__main__":
    # test_json_schema()
    test_no_params_schema()