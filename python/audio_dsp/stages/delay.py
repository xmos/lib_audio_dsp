"""Implementation of delay stage."""

from typing import Optional

from audio_dsp.models.delay_model import DelayParameters
from audio_dsp.design.stage import Stage, StageOutputList
from audio_dsp.stages.signal_chain import Delay as SignalChainDelay


class DelayStage(SignalChainDelay):
    """Stage implementation of delay effect."""

    def __init__(
        self,
        max_delay: Optional[int] = None,
        starting_delay: Optional[int] = None,
        units: str = "samples",
        config: Optional[dict] = None,
        parameters: Optional[dict] = None,
        **kwargs
    ):
        """Initialize delay stage.
        
        Can be initialized either directly with parameters or with config/parameters dicts.
        
        Args:
            max_delay: Maximum delay length in samples
            starting_delay: Initial delay length in samples
            units: Units for delay values, either "samples" or "seconds"
            config: Configuration dictionary containing max_delay
            parameters: Parameters dictionary containing delay and units
            **kwargs: Additional arguments passed to parent class
        """
        # Get config values
        if config is not None:
            max_delay = config.get("max_delay", max_delay)
            
        # Get parameter values
        if parameters is not None:
            starting_delay = parameters.get("delay", starting_delay)
            units = parameters.get("units", units)
            
        # Ensure we have valid values
        max_delay = max_delay if max_delay is not None else 1024
        starting_delay = starting_delay if starting_delay is not None else 0
            
        super().__init__(
            max_delay=max_delay,
            starting_delay=starting_delay,
            units=units,
            **kwargs
        )
        
        # Store parameters
        self.max_delay = max_delay
        self.parameters = DelayParameters(delay=starting_delay, units=units)

    def set_parameters(self, parameters: DelayParameters):
        """Set delay parameters.
        
        Args:
            parameters: New delay parameters to apply
        """
        if parameters.delay > self.max_delay:
            raise ValueError(
                f"Delay value {parameters.delay} exceeds maximum "
                f"delay {self.max_delay}"
            )
            
        self.set_delay(parameters.delay, units=parameters.units)
        self.parameters = parameters 