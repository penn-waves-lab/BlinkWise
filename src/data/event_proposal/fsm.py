from functools import partial
from typing import Callable, Optional

from .buffer_manager import RFFrameDualBuffer


class State:
    """
    a state in finite state machine.
    """

    def __init__(self, name):
        self.name = name
        self.next_possible_transitions: list[Transition] = []

    def add_transition(self, transition):
        self.next_possible_transitions.append(transition)

    def __repr__(self):
        return f"State: {self.name}"


class Transition:
    """
    a class representing transition between states
    """

    def __init__(self, to_state: str, condition: Callable[[int], bool], action: Callable = None):
        self.to_state = to_state
        self.condition = condition
        self.action = action

    def is_condition_met(self, label):
        return self.condition(label)

    def act(self, new_batch):
        if self.action is None:
            return

        return self.action(new_batch)

    def __repr__(self):
        return f"Transition: cond - {self.condition.__name__}; to - {self.to_state}"


class FiniteStateMachine:
    """
    a general finite state machine.
    """

    def __init__(self):
        self.states: list[State] = []
        self.state_names: dict[str, int] = {}

        self.current_state_idx: Optional[int] = None

    @property
    def current_state(self):
        if self.current_state_idx is None:
            raise ValueError("Accessing current state before setting start state. Use set_start().")
        return self.states[self.current_state_idx]

    def add_state(self, state: State):
        self.states.append(state)

        if state.name in self.state_names.keys():
            raise ValueError(f"Trying to add duplicate state name {state.name}. State names should be unique.")
        self.state_names[state.name] = len(self.states) - 1

    def set_state(self, state_name: str):
        state_index = self.state_names.get(state_name, -1)
        if state_index == -1:
            raise ValueError(f"Invalid state name {state_name}. Have you added this state?")
        self.current_state_idx = state_index

    def set_start(self, state_name: str):
        """
        a wrapper for setting initial state
        """
        self.set_state(state_name)

    def trigger(self, label, new_batch):
        output = None
        for t in self.current_state.next_possible_transitions:
            # only one transition should be triggered
            if t.is_condition_met(label):
                output = t.act(new_batch)
                self.set_state(t.to_state)
        return output


# define condition check functions
def is_one(label) -> bool:
    return label == 1


def is_zero(label) -> bool:
    return label == 0


def prepare_i2a_states(idle_state: State, active_state: State, buffer_manager: RFFrameDualBuffer, i2a_n_pos: int,
                       i2a_n_neg: int) -> dict[str, State]:
    """
    ┌────┐                                          ┌──────┐
    │idle│                                          │active│
    └▲──┬┘   ┌───────────┐        ┌──────────────┐  └─▲────┘
     │  │ 1  │i2a_p_1_n_0│   ...  │i2a_p_N1-1_n_0│ 1  │
     │  └────▼───────────┘        └──────────────┴────┤
     │                                                │
     │             .                     .            │
     │             .                     .            │
     │             .                     .            │
     │                                                │
     │   0   ┌────────────┐       ┌───────────────┐ 1 │
     ├───────┤i2a_p_1_n_N0│  ...  │i2a_p_N1-1_n_N0├───┘
     │       └────────────┘       └───────┬───────┘
     │   0                                │
     └────────────────────────────────────┘
    """

    def idle_2_intermediate(new_batch):
        buffer_manager.transfer_between_buffers("idle", "intermediate")

        # print("idle_2_intermediate", buffer_manager.get_buffer_item_count(["intermediate"])[0])

        buffer_manager.add_to_buffer(["idle", "intermediate"], new_batch)
        return None

    def idle_2_active(new_batch):
        # no intermediate states are defined
        left_margin = buffer_manager.get_buffer_item_count(["idle"])[0]
        buffer_manager.transfer_between_buffers("idle", "target")
        buffer_manager.add_to_buffer(["idle", "target"], new_batch)
        return ([-1] * left_margin) + [1]

    def intermediate_2_intermediate(new_batch):
        buffer_manager.add_to_buffer(["idle", "intermediate"], new_batch)
        return None

    def intermediate_2_active(intermediate_batch_count, new_batch):
        left_margin = buffer_manager.get_buffer_item_count(["intermediate"])[0] - intermediate_batch_count

        # print(left_margin)

        buffer_manager.transfer_between_buffers("intermediate", "target")
        buffer_manager.clear_buffers(["intermediate"])
        buffer_manager.add_to_buffer(["idle", "target"], new_batch)
        return [-left_margin] + ([1] * intermediate_batch_count) + [1]

    def intermediate_2_idle(intermediate_batch_count, new_batch):
        buffer_manager.clear_buffers(["intermediate"])
        buffer_manager.add_to_buffer(["idle"], new_batch)
        return [0] * intermediate_batch_count + [0]

    # initialize all states between transition from idle to active state
    i2a_states = {}
    for p in range(1, i2a_n_pos):
        for n in range(i2a_n_neg):
            state_name = f"i2a_p_{p}_n_{n}"
            i2a_states[state_name] = State(state_name)

    # idle to active
    if i2a_n_pos == 1:
        # immediately transit from the idle to active state when a positive sample is received
        # the lambda function executes the function and return the value at the 2nd position
        idle_state.add_transition(Transition(
            to_state=active_state.name,
            condition=is_one,
            action=idle_2_active
        ))
    else:
        # more positive samples are needed to transit from the idle to active state
        idle_state.add_transition(Transition(
            to_state=i2a_states["i2a_p_1_n_0"].name,
            condition=is_one,
            action=idle_2_intermediate
        ))

        # i2a states that are not on boundaries with idle and active
        for p in range(1, i2a_n_pos):
            for n in range(i2a_n_neg):
                # a positive sample is received
                if i2a_states.get(f"i2a_p_{p + 1}_n_{n}", None) is not None:
                    # there exists state to accept more positive samples while remain intermediate
                    i2a_states[f"i2a_p_{p}_n_{n}"].add_transition(Transition(
                        to_state=i2a_states[f"i2a_p_{p + 1}_n_{n}"].name,
                        condition=is_one,
                        action=intermediate_2_intermediate
                    ))
                # a negative sample is received
                if i2a_states.get(f"i2a_p_{p}_n_{n + 1}", None) is not None:
                    # there exists state to accept more negative samples while remain intermediate
                    i2a_states[f"i2a_p_{p}_n_{n}"].add_transition(Transition(
                        to_state=i2a_states[f"i2a_p_{p}_n_{n + 1}"].name,
                        condition=is_zero,
                        action=intermediate_2_intermediate
                    ))

        # processing boundary i2a states
        #   left-most states connected to active
        for n in range(i2a_n_neg):
            i2a_states[f"i2a_p_{i2a_n_pos - 1}_n_{n}"].add_transition(Transition(
                to_state=active_state.name,
                condition=is_one,
                action=partial(intermediate_2_active, i2a_n_pos - 1 + n)
            ))
        #   bottom states connected to idle
        for p in range(1, i2a_n_pos):
            i2a_states[f"i2a_p_{p}_n_{i2a_n_neg - 1}"].add_transition(Transition(
                to_state=idle_state.name,
                condition=is_zero,
                action=partial(intermediate_2_idle, i2a_n_neg - 1 + p)
            ))

    return i2a_states


def prepare_a2i_states(idle_state: State, active_state: State, buffer_manager: RFFrameDualBuffer, a2i_n_pos: int,
                       a2i_n_neg: int) -> dict[str, State]:
    """
                                                       1
                      ┌────────────────────────────────────┐
                      │                                    │
             ┌────────┴────────┐     ┌──────────────┐  1   │
        ┌────┤a2i_p_M1-1_n_M0-1│ ... │a2i_p_M1-1_n_1├──────┤
        │ 0  └─────────────────┘     └──────────────┘      │
        │                                                  │
        │            .                       .             │
        │            .                       .             │
        │            .                       .             │
        │  0                                         0     │
        ├────┬──────────────┐           ┌───────────▲──┐   │
        │    │a2i_p_0_n_M0-1│    ...    │a2i_p_0_n_1│  │   │
    ┌───▼┐   └──────────────┘           └───────────┘ ┌┴───▼─┐
    │idle│                                            │active│
    └────┘                                            └──────┘
    """

    def reach_idle(new_batch):
        buffer_manager.add_to_buffer(["idle", "target"], new_batch)
        buffer_manager.clear_buffers(["target"])
        return 1

    def add_to_idle_and_target(new_batch):
        buffer_manager.add_to_buffer(["idle", "target"], new_batch)
        return 1

    # initialize all states between transition from active to idle state
    a2i_states = {}
    for p in range(a2i_n_pos):
        for n in range(1, a2i_n_neg):
            state_name = f"a2i_p_{p}_n_{n}"
            a2i_states[state_name] = State(state_name)

    # active to idle
    if a2i_n_neg == 1:
        # immediately transit from the active to idle state when a negative sample is received
        active_state.add_transition(Transition(
            to_state=idle_state.name,
            condition=is_zero,
            action=reach_idle
        ))
    else:
        # more negative samples are needed to transit from the active to idle state
        active_state.add_transition(Transition(
            to_state=a2i_states["a2i_p_0_n_1"].name,
            condition=is_zero,
            action=add_to_idle_and_target
        ))

        # a2i states that are not on boundaries with idle and active
        for p in range(a2i_n_pos):
            for n in range(1, a2i_n_neg):
                # a positive sample is received
                if a2i_states.get(f"a2i_p_{p + 1}_n_{n}", None) is not None:
                    a2i_states[f"a2i_p_{p}_n_{n}"].add_transition(Transition(
                        to_state=a2i_states[f"a2i_p_{p + 1}_n_{n}"].name,
                        condition=is_one,
                        action=add_to_idle_and_target
                    ))
                # a negative sample is received
                if a2i_states.get(f"a2i_p_{p}_n_{n + 1}", None) is not None:
                    a2i_states[f"a2i_p_{p}_n_{n}"].add_transition(Transition(
                        to_state=a2i_states[f"a2i_p_{p}_n_{n + 1}"].name,
                        condition=is_zero,
                        action=add_to_idle_and_target
                    ))

        # processing boundary a2i states
        #   top states connected to active
        for n in range(1, a2i_n_neg):
            a2i_states[f"a2i_p_{a2i_n_pos - 1}_n_{n}"].add_transition(Transition(
                to_state=active_state.name,
                condition=is_one,
                action=add_to_idle_and_target
            ))
        #   left-most states connected to idle
        for p in range(a2i_n_pos):
            a2i_states[f"a2i_p_{p}_n_{a2i_n_neg - 1}"].add_transition(Transition(
                to_state=idle_state.name,
                condition=is_zero,
                action=reach_idle
            ))

    return a2i_states


def get_buffer_fsm(buffer_manager: RFFrameDualBuffer, i2a_n_pos: int, i2a_n_neg: int, a2i_n_pos: int, a2i_n_neg: int):
    """
    Get a finite state machine for frame data selection.

    Args:
        buffer_manager: Class managing frame data buffering operations.
        i2a_n_pos: The number of positive samples to transfer from idle to active state.
        i2a_n_neg: The number of negative samples to transfer from intermediate states from idle to active states back
            to idle state.
        a2i_n_pos: The number of positive samples to transfer from intermediate states from active to idle states back
            to active state.
        a2i_n_neg: The number of negative samples to transfer from idle to active state.

    Returns:
        FiniteStateMachine object representing the configured finite state machine.

    Note:
        The complete state transition diagram:
                                                             1
                            ┌────────────────────────────────────┐
                            │                                    │
                   ┌────────┴────────┐     ┌──────────────┐  1   │
              ┌────┤a2i_p_M1-1_n_M0-1│ ... │a2i_p_M1-1_n_1├──────┤
              │ 0  └─────────────────┘     └──────────────┘      │
              │                                                  │
              │            .                       .             │
              │            .                       .             │
              │            .                       .             │
         0    │  0                                         0     │
        ┌──┐  ├────┬──────────────┐           ┌───────────▲──┐   │
        │  │  │    │a2i_p_0_n_M0-1│    ...    │a2i_p_0_n_1│  │   │
        │ ┌▼──▼┐   └──────────────┘           └───────────┘ ┌┴───▼─┐
        └─┤idle│                                            │active├─┐
          └▲──┬┘   ┌───────────┐           ┌──────────────┐ └▲───▲─┘ │
           │  │ 1  │i2a_p_1_n_0│   ...     │i2a_p_N1-1_n_0│1 │   │ 1 │
           │  └────▼───────────┘           └──────────────┴──┤   └───┘
           │                                                 │
           │             .                                   │
           │             .                                   │
           │             .                                   │
           │                                                 │
           │   0   ┌──────────────┐     ┌─────────────────┐1 │
           ├───────┤i2a_p_1_n_N0-1│...  │i2a_p_N1-1_n_N0-1├──┘
           │       └──────────────┘     └───────┬─────────┘
           │   0                                │
           └────────────────────────────────────┘

        idle self-loop:
            batches are unrelated.
            action: add_to_buffer(["idle"])
            output: 0

        leave idle to intermediate/i2a states and during intermediate states:
            batches are potentially related.
            action: transfer_between_buffer("idle", "intermediate") (when leave); add_to_buffer(["idle", "intermediate"])
            output: None, pending state transition to idle or active.

        reach active from intermediate states:
            all potentially related batches are confirmed related. contents of the intermediate buffer are transferred to the target buffer.
            action: transfer_between_buffer("intermediate", "target"); clear_buffers(["intermediate"]); add_to_buffer(["idle", "target"])
            output: [1] for each cached batch, derived from the last state's name. Several [-1]s appended to indicate conversion for margin batches added to the full buffer.

        return to idle from intermediate states:
            all potentially related batches are rejected. no transfer to full buffer.
            action: clear_buffers(["intermediate"]); add_to_buffer(["idle"])
            output: [0] for each cached batch, derived from the last state's name.

        active self-loop, from active to idle, and reach idle/active:
            all batches are related. right margin is contained in this step.
            action: add_to_buffer(["idle", "target"]); clear_buffer(["target"]) (when reaches idle).
            output: 1

        The size of the small FIFO buffer should include margin batches plus (i2a_n_neg - 1) + i2a_n_pos.
        The right margin ranges from minimal a2i_n_neg to maximal a2i_n_neg + (a2i_n_pos - 1).
        Outputs are used for parameter searching and can be compared with ground truth labeling.
    """

    if i2a_n_pos < 1:
        raise ValueError("At least 1 positive sample needed to transit from idle to active state.")
    if i2a_n_neg < 1:
        raise ValueError("At least 1 negative sample needed to transit from i2a states to idle state.")

    if a2i_n_neg < 1:
        raise ValueError("At least 1 negative sample needed to transit from active to idle state.")
    if a2i_n_pos < 1:
        raise ValueError("At least 1 positive sample needed to transit from a2i states to idle state.")

    # add states to the FSM
    idle_state = State("idle")
    active_state = State("active")
    i2a_states = prepare_i2a_states(idle_state, active_state, buffer_manager, i2a_n_pos=i2a_n_pos, i2a_n_neg=i2a_n_neg)
    a2i_states = prepare_a2i_states(idle_state, active_state, buffer_manager, a2i_n_pos=a2i_n_pos, a2i_n_neg=a2i_n_neg)

    # self loop
    idle_state.add_transition(Transition(
        to_state=idle_state.name,
        condition=is_zero,
        action=lambda new_batch: (buffer_manager.add_to_buffer(["idle"], new_batch), 0)[1]
    ))
    active_state.add_transition(Transition(
        to_state=active_state.name,
        condition=is_one,
        action=lambda new_batch: (buffer_manager.add_to_buffer(["idle", "target"], new_batch), 1)[1]
    ))

    # construct the finite state machine
    fsm = FiniteStateMachine()
    fsm.add_state(active_state)
    fsm.add_state(idle_state)
    for s in i2a_states.values():
        fsm.add_state(s)
    for s in a2i_states.values():
        fsm.add_state(s)
    fsm.set_start(idle_state.name)

    return fsm
