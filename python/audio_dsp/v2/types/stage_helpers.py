from . import UnknownEdge


def outputs_match_inputs(context, inputs, n_out, allowable_input_types=None):
    """
    Determine outputs for a stage who's outputs have the same type as the inputs. Fails
    if inputs dont all have the same type

    n_out: number of outputs
    allowable_input_types: list of strings of type names to check
    """
    valid_inputs = [i for i in inputs if not isinstance(i, UnknownEdge)]
    if allowable_input_types and not all(i.name in allowable_input_types for i in valid_inputs):
        raise ValueError(f"Expected one of {allowable_input_types}")
    if valid_inputs:
        n = valid_inputs[0].shape
        if not all(i.shape == n for i in valid_inputs):
            raise ValueError("Input shapes do not match")

        # produce outputs
        return [valid_inputs[0].model_copy() for _ in range(n_out)]
    return [UnknownEdge()] * n_out
