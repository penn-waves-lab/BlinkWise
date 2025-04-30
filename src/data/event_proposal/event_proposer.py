from collections import deque

import numpy as np

from .buffer_manager import RFFrameDualBuffer
from .fsm import get_buffer_fsm


class EventProposer:
    """
    Event proposer is a state machine that detects positive frames in a sequence of data.
    """
    def __init__(
            self,
            # buffer manager
            batch_shape,
            idle_buffer_n_batches: int,
            # fsm
            i2a_n_pos: int,
            i2a_n_neg: int,
            a2i_n_pos: int,
            a2i_n_neg: int,
    ) -> None:
        """
        Initialize the event proposer.

        Please refer to the documentation of ``RFFrameDualBuffer`` and ``get_buffer_fsm`` for meaning of the parameters.
        """

        # buffer manager
        self.batch_shape = batch_shape
        self.idle_buffer_n_batches = idle_buffer_n_batches
        # all buffers use the dummy buffer.
        self.dual_buffer = RFFrameDualBuffer(
            batch_shape=batch_shape,
            idle_buffer_n_batches=-self.idle_buffer_n_batches,
            intermediate_buffer_n_batches=-1,
            target_buffer_n_batches=-1
        )
        # fsm
        self.i2a_n_pos = i2a_n_pos
        self.i2a_n_neg = i2a_n_neg
        self.a2i_n_pos = a2i_n_pos
        self.a2i_n_neg = a2i_n_neg
        self.machine = get_buffer_fsm(
            self.dual_buffer,
            i2a_n_pos=self.i2a_n_pos,
            i2a_n_neg=self.i2a_n_neg,
            a2i_n_pos=self.a2i_n_pos,
            a2i_n_neg=self.a2i_n_neg
        )

    def detect(self, is_outlier):
        """
        Apply observations of blink patterns as FSM rules to detect clusters of positive frames.

        Args:
            is_outlier: A binary array of shape (n_frames,) where 1 indicates a positive frame and 0 indicates a negative frame.

        Returns:
            A binary array indicating the detected positive frames after applying the FSM rules.
        """
        outputs = deque([])
        dummy_data = np.random.random(self.batch_shape)
        for i in is_outlier:
            o = self.machine.trigger(i, dummy_data)
            if o is None:
                continue
            elif o == 0 or o == 1:
                outputs.append(o)
            elif isinstance(o, list):
                if o[0] < 0:
                    to_pop = abs(o[0])
                    for _ in range(to_pop):
                        outputs.pop()
                    outputs.extend([1] * to_pop)
                    outputs.extend(o[1:])
                else:
                    outputs.extend(o)

        # terminate the state machine by flooding more 0s if it is not idle.
        while not self.machine.current_state.name == "idle":
            o = self.machine.trigger(0, dummy_data)
            if o is None:
                continue
            elif o == 0 or o == 1:
                outputs.append(o)
            elif isinstance(o, list):
                if o[0] < 0:
                    to_pop = abs(o[0])
                    for _ in range(to_pop):
                        outputs.pop()
                    outputs.extend([1] * to_pop)
                    outputs.extend(o[1:])
                else:
                    outputs.extend(o)

        return np.array(outputs)[:is_outlier.shape[0]]
