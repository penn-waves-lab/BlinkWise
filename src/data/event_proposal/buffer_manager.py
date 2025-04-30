from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
from scipy import signal


# =========================
# buffer classes
# =========================

class Buffer(ABC):
    """
    An abstract class for buffer managers.
    """
    def __init__(self, n_batches, batch_shape):
        self.n_batches: int = int(n_batches)
        self.n_frames: int = int(batch_shape[0])

        self.max_length: int = int(self.n_batches * self.n_frames)
        self.batch_shape = batch_shape

    @abstractmethod
    def is_empty(self) -> bool:
        pass

    @abstractmethod
    def to_array(self) -> np.ndarray:
        pass

    @abstractmethod
    def add(self, new_batch: np.ndarray) -> None:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass

    @property
    @abstractmethod
    def item_count(self) -> int:
        pass


class FIFOBuffer(Buffer):
    """
    A buffer that stores the last n_batches of frames.
    """
    def __init__(self, n_batches, batch_shape):
        super().__init__(n_batches, batch_shape)

        # -1 means the buffer is empty
        self.all_used = False
        self.head_ptr: int = -1
        self.buffer = np.zeros((self.max_length,) + tuple(self.batch_shape)[1:])

    def __str__(self):
        return "head_ptr: " + str(self.head_ptr) + "\narray: " + str(self.buffer) + "\nrolled: " + str(self.to_array())

    def is_empty(self) -> bool:
        return self.head_ptr == -1

    def to_array(self) -> np.ndarray:
        if not self.all_used:
            return self.buffer[:(self.head_ptr + 1) * self.n_frames]

        batch_to_rotate = (self.head_ptr + 1) % self.n_batches
        return np.roll(self.buffer, -batch_to_rotate * self.n_frames, axis=0)  # this will force a copy

    def add(self, new_batch: np.ndarray) -> None:
        next_ptr = self.head_ptr + 1
        if not self.all_used and next_ptr >= self.n_batches:
            self.all_used = True
        self.head_ptr = next_ptr % self.n_batches

        batch_indices = self.head_ptr * self.n_frames + np.arange(self.n_frames)
        self.buffer[batch_indices] = new_batch

    def clear(self):
        self.all_used = False
        self.head_ptr: int = -1
        self.buffer = np.zeros((self.max_length,) + tuple(self.batch_shape)[1:])

    @property
    def item_count(self) -> int:
        if self.all_used:
            return self.n_batches
        else:
            return self.head_ptr + 1


class AutoDownSamplingBuffer(Buffer):
    """
    A buffer that automatically down-samples incoming data when it is full.

    Deprecated:
        This class is not used in the current implementation.
    """
    def __init__(self, n_batches, batch_shape):
        super().__init__(n_batches, batch_shape)

        # counting how many batches are stored in this buffer. can be larger that n_batch
        self.current_size = 0
        self.buffer = np.zeros((self.max_length,) + tuple(self.batch_shape)[1:])

    def is_empty(self) -> bool:
        return self.current_size == 0

    def to_array(self) -> np.ndarray:
        # Return the non-empty part of the buffer
        if self.current_size < self.n_batches:
            valid_indices = (self.current_size) * self.n_frames
            return self.buffer[:valid_indices]
        else:
            return self.buffer

    def add(self, new_batch: np.ndarray) -> None:
        if self.current_size < self.n_batches:
            batch_indices = self.current_size * self.n_frames + np.arange(self.n_frames)
            self.buffer[batch_indices] = new_batch
            self.current_size += 1
        else:
            downsampling_ratio = self.current_size / (self.current_size + 1)
            n_old_data = min(self.max_length - 1, round(downsampling_ratio * self.max_length))
            n_new_data = self.max_length - n_old_data

            self.buffer[:n_old_data] = signal.resample(self.buffer, n_old_data, axis=0)
            self.buffer[n_old_data:] = signal.resample(new_batch, n_new_data, axis=0)

            self.current_size += 1

    def clear(self):
        # Remove all frames in the buffer
        self.current_size = 0
        self.buffer = np.zeros((self.max_length,) + tuple(self.batch_shape)[1:])

    @property
    def item_count(self) -> int:
        return self.current_size


class DummyBuffer(Buffer):
    """
    A dummy buffer that does not store any data. Suitable for our case where only labels are needed.
    """
    def __init__(self, n_batches, batch_shape, circular: bool = False, verbose: bool = True):
        super().__init__(n_batches, batch_shape)
        self.size = 0
        self.circular = circular
        self.verbose = verbose

    def add(self, new_batch: np.ndarray) -> None:
        if self.circular:
            self.size = min(self.n_batches, self.size + 1)
        else:
            self.size += 1
        if self.verbose:
            print(f"Item put into queue, new size: {self.size}")

    def to_array(self) -> np.ndarray:
        if self.verbose:
            if self.size == 0:
                print("Queue is empty, nothing to get.")
            else:
                self.size -= 1
                print(f"Item retrieved from queue, new size: {self.size}")
        return np.array([np.nan])

    def is_empty(self) -> bool:
        return self.size == 0

    def clear(self):
        self.size = 0

    @property
    def item_count(self) -> int:
        return self.size


class RFFrameDualBuffer:
    """
    using two buffers to manage incoming frames.

    idle_buffer is a FIFO buffer, which always maintain a small window of frames.
    intermediate_buffer is a for hosting transition between idle to active.
    target_buffer is an auto-resizing buffer, which automatically starts to resize incoming data when it is full.
        after recurrentization, the full_buffer will become a queue.
    """

    def __init__(
            self,
            batch_shape,
            idle_buffer_n_batches: int,
            intermediate_buffer_n_batches: int,
            target_buffer_n_batches: int = 100
    ):
        if 0 < intermediate_buffer_n_batches < idle_buffer_n_batches:
            raise ValueError("intermediate_buffer_n_batches must be no smaller than idle_buffer_n_batches.")

        if 0 < target_buffer_n_batches < idle_buffer_n_batches:
            raise ValueError("target_buffer_n_batches must be no smaller than idle_buffer_n_batches.")

        self.batch_shape = batch_shape
        self.n_frames = self.batch_shape[0]

        # setting up the idle buffer
        self.idle_buffer_n_batches = idle_buffer_n_batches
        if self.idle_buffer_n_batches < 0:
            self.idle_buffer = DummyBuffer(abs(self.idle_buffer_n_batches), self.batch_shape, circular=True,
                                           verbose=False)
        else:
            self.idle_buffer = FIFOBuffer(self.idle_buffer_n_batches, self.batch_shape)

        # setting up the intermediate buffer
        self.intermediate_buffer_n_batches = intermediate_buffer_n_batches
        if self.intermediate_buffer_n_batches == -1:
            self.intermediate_buffer = DummyBuffer(self.intermediate_buffer_n_batches, self.batch_shape, verbose=False)
        else:
            self.intermediate_buffer = FIFOBuffer(self.intermediate_buffer_n_batches, self.batch_shape)

        # setting up the target buffer
        self.target_buffer_n_batches = target_buffer_n_batches
        if self.target_buffer_n_batches == -1:
            self.target_buffer = DummyBuffer(self.target_buffer_n_batches, self.batch_shape, verbose=False)
        else:
            self.target_buffer = AutoDownSamplingBuffer(self.target_buffer_n_batches, self.batch_shape)

    def _get_buffer(self, buffer_name: Literal["idle", "intermediate", "target"]) -> Buffer:
        if buffer_name == "idle":
            return self.idle_buffer
        elif buffer_name == "intermediate":
            return self.intermediate_buffer
        elif buffer_name == "target":
            return self.target_buffer
        else:
            raise ValueError(f"Unknown buffer name {buffer_name}. Supported: idle, intermediate, and target.")

    def add_to_buffer(self, buffer_names: list[Literal["idle", "intermediate", "target"]], new_batch):
        buffer_names = list(set(buffer_names))
        buffers = [self._get_buffer(bn) for bn in buffer_names]
        for b in buffers:
            b.add(new_batch)

    def clear_buffers(self, buffer_names: list[Literal["idle", "intermediate", "target"]]):
        buffer_names = list(set(buffer_names))
        buffers = [self._get_buffer(bn) for bn in buffer_names]
        for b in buffers:
            b.clear()

    def transfer_between_buffers(self, source_name, target_name):
        to_transfer = self._get_buffer(source_name).to_array()
        if np.all(np.isnan(to_transfer)):
            # in case of dummy buffer.
            # no real data is needed if the target is also a dummy buffer.
            # else we should skip.
            if self._get_buffer(target_name).__class__.__name__ == "DummyBuffer":
                for i in range(self._get_buffer(source_name).item_count):
                    self.add_to_buffer([target_name], to_transfer)
            else:
                return
        else:
            to_transfer_batch = to_transfer.shape[0] // self.n_frames
            for i in range(to_transfer_batch):
                self.add_to_buffer([target_name], to_transfer[i * self.n_frames: (i + 1) * self.n_frames])

    def get_buffer_item_count(self, buffer_names: list[Literal["idle", "intermediate", "target"]]):
        buffer_names = list(set(buffer_names))
        item_counts = [self._get_buffer(bn).item_count for bn in buffer_names]
        return item_counts
