import math
from dataclasses import dataclass


@dataclass
class BlinkDefinitions:
    """
    Definitions for the blink labels and corresponding colors and text descriptions
    """
    NON_BLINKING = 0
    CLOSING = 1
    INTERPHASE = 2
    REOPENING = 3
    UNKNOWN = 4
    NOT_INITIALIZED = -1
    # events that are detected by the event proposal algorithm but are not blinks, like eyeball movements
    NON_BLINK_EVENT = -2


    # Color mappings
    LABEL_COLORS = {
        NON_BLINKING: "#93BEDF",
        CLOSING: "#8EF9F3",
        INTERPHASE: "#8377D1",
        REOPENING: "#5EFC8D",
        NOT_INITIALIZED: "#FFBFB0",
        UNKNOWN: "#DBD4D3"
    }

    # Text mappings
    LABEL_TEXTS = {
        NON_BLINKING: "non_blinking",
        CLOSING: "closing",
        INTERPHASE: "interphase",
        REOPENING: "reopening",
        NON_BLINK_EVENT: "non_blink_event"
    }

    @staticmethod
    def label_to_text(label: int) -> str:
        """
        Convert label to text description

        Args:
            label: label to convert

        Returns:
            str: text description of the label
        """
        return BlinkDefinitions.LABEL_TEXTS.get(label, "UNKNOWN")

    def text_to_label(self, label_text: str) -> int:
        """
        Convert text description to label

        Args:
            label_text: text description to convert

        Returns:
            int: label of the text description
        """
        inverse_mapping = {v: k for k, v in BlinkDefinitions.LABEL_TEXTS.items()}
        inverse_mapping["non-blinking"] = self.NON_BLINKING
        inverse_mapping["non-blink-event"] = self.NON_BLINK_EVENT
        return inverse_mapping.get(label_text.strip().lower(), self.UNKNOWN)


@dataclass
class Constants:
    """
    Constants used in the project
    """
    ROUNDED_FPS: int = 480 # slo-mo video fps, rounded to nearest integer
    RADAR_FRAME_INTERVAL: float = 0.0011111111380159855
    LED_BLINK_INTERVAL: float = 10.02  # seconds
    LED_ON_DURATION: float = 0.01  # seconds

    DEFAULT_VIDEO_FPS_DENOMINATOR: float = 1.001281950671856
    # the real fps is slightly below 480 after LED synchronization, so we use this value to correct the video fps
    LED_ON_DURATION_IN_FRAMES: int = math.floor(LED_ON_DURATION * ROUNDED_FPS)
    RADAR_FPS: float = 1 / RADAR_FRAME_INTERVAL
    DEFAULT_CORRECTED_VIDEO_FPS: float = ROUNDED_FPS / DEFAULT_VIDEO_FPS_DENOMINATOR
    # corrected video fps after LED synchronization


blink_defs = BlinkDefinitions()
constants = Constants()
