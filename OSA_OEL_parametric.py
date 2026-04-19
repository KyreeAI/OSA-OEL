"""
Parametric OSA‑OEL Implementation
================================

This module contains a refactored version of the Oscillatory
Synthesizer Algorithm (OSA‑OEL) that emphasises configurability
through parameter injection.  In the original design the core
components—scale tuning, harmonic frequency selection, graph based
relationship modelling and memory handling—were tightly
intertwined.  This refactor exposes clean interfaces for each
component and introduces a small data‐class to describe tuners and
their parameters.  A configuration file (JSON) can be used to
define an arbitrary number of tuners up front; these tuners are then
executed by a `ParametricSynthesizer` which orchestrates the
computation.

Key features
------------

* **Declarative tuner definition.**  Each tuner is described with
  a `TunerConfig` object which records the tuner name, the type of
  frequency generation required (`sine`, `cosine`, `random` etc.)
  and a dictionary of parameters specific to that tuner.

* **Dynamic configuration loading.**  The synthesizer can be
  instantiated from a JSON configuration file.  This decouples
  algorithmic logic from configuration data and makes it easy to
  experiment with new tuners by simply editing a JSON file.

* **Reusable core components.**  Frequency tuning and harmonic
  generation are encapsulated in `ResonantScaleModel` and
  `HarmonicFrequencySelector` respectively.  Graph relationships
  live in `WeightedGraph`, while reinforcement and short term
  memory are provided by `RLMemory` and `ShortTermMemory`.

* **Extensible synthesizer.**  The `ParametricSynthesizer` exposes
  a simple `run` method that iterates over the loaded tuners and
  dispatches to the appropriate frequency generator.  New
  frequency types can be added by extending the `_dispatch` method.

This code is intended as a starting point for further research
into OSA‑OEL.  You can run the synthesizer with your own
configuration like so:

```python
from OSA_OEL_parametric import (ResonantScaleModel, HarmonicFrequencySelector,
                                ParametricSynthesizer)

# create core objects
resonant = ResonantScaleModel(tuning_standard=440.0)
selector = HarmonicFrequencySelector(resonant)

# load tuners from a JSON file
synth = ParametricSynthesizer.from_json(resonant, selector, "config.json")

# produce the frequencies for each tuner
results = synth.run()
for name, freqs in results.items():
    print(name, freqs)
```

"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np


@dataclass
class TunerConfig:
    """Dataclass capturing configuration for a single tuner.

    Attributes
    ----------
    name:
        Unique identifier for the tuner.  This is used as the key
        in synthesizer results.
    frequency_type:
        The generation method to use.  Supported values include
        ``sine``, ``cosine`` and ``random``.  See
        :class:`ParametricSynthesizer` for details on how each type
        is handled.
    params:
        A freeform dictionary of parameters passed directly to the
        generation function.  For example, you might provide
        ``{"note_index": 0, "octave": 1, "scale_type": "minor"}``.
    """

    name: str
    frequency_type: str
    params: Dict[str, Any] = field(default_factory=dict)


class ResonantScaleModel:
    """Generate frequencies based on musical scales and tuning.

    The resonant scale model defines a mapping from note indices and
    octaves into actual frequencies (in hertz).  It supports a
    variety of western scales and can be extended to microtonal
    systems by modifying the internal `scales` dictionary.

    Parameters
    ----------
    tuning_standard:
        The base frequency for ``A4``.  The default value of 440.0
        Hz yields the familiar concert pitch.  Alternative tuning
        systems such as 432 Hz can be specified here.
    """

    def __init__(self, tuning_standard: float = 440.0) -> None:
        self.tuning_standard = tuning_standard
        # Definition of common scales: major, natural minor and
        # harmonic minor.  Each scale is described by the semitone
        # offsets from the root note.
        self.scales: Dict[str, List[int]] = {
            "major": [2, 2, 1, 2, 2, 2, 1],
            "natural_minor": [2, 1, 2, 2, 1, 2, 2],
            "harmonic_minor": [2, 1, 2, 2, 1, 3, 1],
        }

    def tune_scale(self,
                   note_index: int,
                   octave: int,
                   scale_type: str = "major") -> float:
        """Return the frequency for a given scale degree.

        Parameters
        ----------
        note_index:
            Zero‑based index indicating which degree of the scale to
            use.  For example, 0 is the root, 1 the second degree and
            so on.
        octave:
            The octave offset relative to the tuning standard.  An
            octave of 0 means the scale starts at the frequency of
            ``A4``; 1 means one octave above, and –1 means one
            octave below.
        scale_type:
            Name of the scale.  Must be one of the keys in
            :attr:`scales`.

        Returns
        -------
        float
            The computed frequency in hertz.

        Notes
        -----
        Frequencies are calculated by cumulatively adding scale
        intervals from the root and transposing up or down by
        octaves.  This method supports any scale structure defined in
        :attr:`scales`.  If a scale is not found, a ValueError is
        raised.
        """
        if scale_type not in self.scales:
            raise ValueError(f"Unknown scale type '{scale_type}'.")

        intervals = self.scales[scale_type]
        # wrap around the intervals if the note index exceeds the
        # length of the scale.  This allows for generating notes
        # across multiple octaves seamlessly.
        semitone_offset = 0
        for i in range(note_index + 1):
            semitone_offset += intervals[i % len(intervals)]

        # compute the base frequency for A4 adjusted by octaves
        base_frequency = self.tuning_standard * (2 ** octave)
        frequency = base_frequency * (2 ** (semitone_offset / 12))
        return frequency


class HarmonicFrequencySelector:
    """Select frequencies based on harmonic relationships.

    This class provides methods to generate frequencies that are
    musically consonant or dissonant relative to a given note.  It
    uses integer ratios derived from the harmonic series as well as
    random selections for exploratory behaviour.
    """

    def __init__(self, resonant_model: ResonantScaleModel) -> None:
        self.resonant_model = resonant_model

    def generate_harmonics(self,
                           root_note_index: int,
                           octave: int,
                           num_harmonics: int = 5) -> List[float]:
        """Generate a list of harmonically related frequencies.

        Harmonics are calculated as integer multiples of the root
        frequency.  For example, the first harmonic (n=1) is the root
        itself, the second harmonic (n=2) is one octave above, and so
        on.
        """
        root_freq = self.resonant_model.tune_scale(root_note_index, octave)
        harmonics = [root_freq * (n + 1) for n in range(num_harmonics)]
        return harmonics

    def generate_random_harmonics(self,
                                  root_note_index: int,
                                  octave: int,
                                  count: int = 5) -> List[float]:
        """Generate a list of pseudo‑random frequencies around a root.

        This method produces `count` frequencies clustered around the
        root frequency.  The random offsets are generated from a
        normal distribution centred at 0 with a small variance to
        prevent wildly divergent frequencies.  Such randomness can
        introduce novelty into OSA's emergent behaviour.
        """
        root_freq = self.resonant_model.tune_scale(root_note_index, octave)
        noise = np.random.normal(loc=0.0, scale=0.05, size=count)
        return [root_freq * (1 + n) for n in noise]


class WeightedGraph:
    """Directed graph with mutable edge weights.

    Nodes are identified by arbitrary hashable objects (strings,
    integers, etc.).  Edges are directed and carry floating‑point
    weights.  The graph can be used to model influence, attention or
    other relationships between tuners.
    """

    def __init__(self) -> None:
        self.adj_list: Dict[Any, Dict[Any, float]] = {}

    def add_node(self, node: Any) -> None:
        if node not in self.adj_list:
            self.adj_list[node] = {}

    def add_edge(self, src: Any, dest: Any, weight: float) -> None:
        self.add_node(src)
        self.add_node(dest)
        self.adj_list[src][dest] = weight

    def update_weight(self, src: Any, dest: Any, delta: float) -> None:
        if src in self.adj_list and dest in self.adj_list[src]:
            self.adj_list[src][dest] += delta

    def get_neighbors(self, node: Any) -> Dict[Any, float]:
        return self.adj_list.get(node, {})

    def __repr__(self) -> str:
        return f"WeightedGraph({self.adj_list})"


class RLMemory:
    """Simple reinforcement learning memory buffer.

    The buffer stores experiences as tuples of (state, action,
    reward, next_state, done).  It provides a mechanism to record
    interactions and sample them later for learning.
    """

    def __init__(self, capacity: int = 10000) -> None:
        self.capacity = capacity
        self.buffer: List[Any] = []

    def add_experience(self,
                        state: Any,
                        action: Any,
                        reward: float,
                        next_state: Any,
                        done: bool) -> None:
        experience = (state, action, reward, next_state, done)
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> List[Any]:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]


class ShortTermMemory:
    """Cache of recent states to preserve short term context.

    This structure holds a fixed number of recent items.  When
    capacity is exceeded, the oldest entry is discarded.  It can be
    used to feed the current context back into the algorithm to
    improve continuity.
    """

    def __init__(self, capacity: int = 5) -> None:
        self.capacity = capacity
        self.buffer: List[Any] = []

    def add(self, item: Any) -> None:
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(item)

    def get_context(self) -> List[Any]:
        return list(self.buffer)


class ParametricSynthesizer:
    """Orchestrates tuner execution based on supplied configuration.

    The synthesizer holds references to core frequency generators and
    applies them in a loop over each configured tuner.  Results are
    returned as a dictionary keyed by the tuner's name.
    """

    def __init__(self,
                 resonant_model: ResonantScaleModel,
                 harmonic_selector: HarmonicFrequencySelector,
                 tuner_configs: Dict[str, TunerConfig]) -> None:
        self.resonant_model = resonant_model
        self.harmonic_selector = harmonic_selector
        self.tuner_configs = tuner_configs

    @classmethod
    def from_json(cls,
                  resonant_model: ResonantScaleModel,
                  harmonic_selector: HarmonicFrequencySelector,
                  json_path: str) -> "ParametricSynthesizer":
        """Construct a synthesizer from a JSON configuration file.

        The JSON file should contain an object where each key is the
        tuner name and the value is an object with keys
        ``frequency_type`` and ``params``.  For example::

            {
              "tuner_A": {
                "frequency_type": "sine",
                "params": {"note_index": 0, "octave": 0, "scale_type": "major"}
              },
              "tuner_B": {
                "frequency_type": "random",
                "params": {"note_index": 2, "octave": 1}
              }
            }

        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        configs: Dict[str, TunerConfig] = {}
        for name, cfg in data.items():
            frequency_type = cfg.get('frequency_type')
            params = cfg.get('params', {})
            configs[name] = TunerConfig(name=name, frequency_type=frequency_type, params=params)
        return cls(resonant_model, harmonic_selector, configs)

    def _dispatch(self, cfg: TunerConfig) -> List[float] | float:
        """Dispatch a tuner to the appropriate generation function.

        This private method centralises the logic for selecting the
        right frequency generator based on the tuner's
        ``frequency_type``.  It returns either a single frequency
        (float) or a list of frequencies depending on the type.
        """
        # Unpack commonly used parameters with sensible defaults.
        note_index = int(cfg.params.get('note_index', 0))
        octave = int(cfg.params.get('octave', 0))
        scale_type = cfg.params.get('scale_type', 'major')
        num_harmonics = int(cfg.params.get('num_harmonics', 5))
        count = int(cfg.params.get('count', 5))

        ft = cfg.frequency_type.lower()
        if ft == 'sine' or ft == 'cosine':
            # For both sine and cosine we return a single scale tuned frequency.
            return self.resonant_model.tune_scale(note_index, octave, scale_type)
        elif ft == 'random':
            return self.harmonic_selector.generate_random_harmonics(note_index, octave, count)
        elif ft == 'harmonic':
            return self.harmonic_selector.generate_harmonics(note_index, octave, num_harmonics)
        else:
            raise ValueError(f"Unsupported frequency type '{cfg.frequency_type}' for tuner '{cfg.name}'.")

    def run(self) -> Dict[str, List[float] | float]:
        """Execute all configured tuners and return their outputs.

        Returns
        -------
        dict
            A dictionary mapping tuner names to their generated
            frequencies.  A value may be either a single float or a
            list of floats, depending on the tuner's `frequency_type`.
        """
        results: Dict[str, List[float] | float] = {}
        for name, cfg in self.tuner_configs.items():
            results[name] = self._dispatch(cfg)
        return results


class ParametricEmbeddings:
    """Utility for encoding and decoding patterns into vector space.

    Embedding strategies can be customised by overriding the
    ``encode`` and ``decode`` methods.  The default implementation
    uses a simple random projection for encoding arbitrary objects and
    a weighted average for combining embeddings.
    """

    def __init__(self, embedding_dim: int = 128) -> None:
        self.embedding_dim = embedding_dim

    def encode(self, item: Any) -> np.ndarray:
        # For demonstration purposes, we hash the item and seed the
        # random generator to produce a deterministic embedding.
        np.random.seed(hash(item) & 0xFFFFFFFF)
        return np.random.normal(size=self.embedding_dim)

    def decode(self, embedding: np.ndarray) -> Any:
        # In a real system this method would perform the inverse
        # operation of encode.  Here we simply return the embedding
        # itself for introspection.
        return embedding

    def combine_embeddings(self, embeddings: Iterable[np.ndarray], weights: Optional[Iterable[float]] = None) -> np.ndarray:
        embeddings = list(embeddings)
        if not embeddings:
            return np.zeros(self.embedding_dim)
        if weights is None:
            weights = [1.0] * len(embeddings)
        weighted_sum = sum(w * emb for w, emb in zip(weights, embeddings))
        total_weight = sum(weights)
        return weighted_sum / total_weight


__all__ = [
    'TunerConfig',
    'ResonantScaleModel',
    'HarmonicFrequencySelector',
    'WeightedGraph',
    'RLMemory',
    'ShortTermMemory',
    'ParametricSynthesizer',
    'ParametricEmbeddings',
]