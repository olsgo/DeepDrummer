import os
import copy
import numpy as np


def categorize_sample(sample_name):
    """Categorize a drum sample based on its filename."""
    name = sample_name.lower()
    if 'kick' in name:
        return 'kick'
    if 'snare' in name or 'snr' in name:
        return 'snare'
    if 'hat' in name or 'hh' in name or 'shaker' in name:
        return 'hihat'
    if 'tom' in name:
        return 'tom'
    if 'cym' in name or 'crash' in name or 'ride' in name:
        return 'cymbal'
    return 'perc'  # Default to general percussion


def apply_velocity_decay(sample, velocity, instrument_category):
    """Apply an exponential decay envelope based on the note velocity."""
    # Short percussive sounds like kicks or snares usually do not need extra
    # decay shaping, so leave them unchanged.
    if instrument_category in ["kick", "snare"]:
        return sample

    # Lower velocities should result in shorter sounds. We square the inverse
    # velocity so that soft hits decay much faster than loud ones.
    decay_strength = 5.0 * (1.0 - velocity) ** 2
    envelope = np.exp(-np.linspace(0.0, decay_strength, num=len(sample)))

    return sample * envelope


class AdvancedGrooveProfile:
    """Timing/velocity adjustments for different instrument categories."""

    def __init__(self, name, description, groove_maps):
        self.name = name
        self.description = description
        self.groove_maps = {}
        for category, data in groove_maps.items():
            self.groove_maps[category] = {
                'timing': np.array(data['timing'], dtype=np.float32),
                'velocity': np.array(data['velocity'], dtype=np.float32),
            }

    def get_groove_for_instrument(self, category):
        if category in self.groove_maps:
            return self.groove_maps[category]
        if 'perc' in self.groove_maps:
            return self.groove_maps['perc']
        return {'timing': np.zeros(16), 'velocity': np.ones(16)}


# ---------------------------------------------------------------------------
# Instrument-specific groove profiles derived from the Groove MIDI Dataset
# ---------------------------------------------------------------------------
GROOVE_ADVANCED_FUNK = AdvancedGrooveProfile(
    name="Advanced Funk",
    description="Swung 16ths on hi-hats, solid kick/snare foundation.",
    groove_maps={
        'kick': {
            'timing': [2, 0, 3, 0, 2, 0, 3, 0, 2, 0, 3, 0, 2, 0, 3, 0],
            'velocity': [1.1, 1.0, 1.1, 1.0, 1.1, 1.0, 1.1, 1.0,
                        1.1, 1.0, 1.1, 1.0, 1.1, 1.0, 1.1, 1.0],
        },
        'snare': {
            'timing': [0, 0, 5, 0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0, 5, 0],
            'velocity': [1.0, 1.0, 1.2, 1.0, 1.0, 1.0, 1.2, 1.0,
                        1.0, 1.0, 1.2, 1.0, 1.0, 1.0, 1.2, 1.0],
        },
        'hihat': {
            'timing': [-0.8, 6.5, -1.2, 10.2, -0.5, 6.8, -1.0, 10.5,
                       -0.7, 6.6, -1.1, 10.3, -0.6, 6.7, -0.9, 10.4],
            'velocity': [0.9, 1.0, 0.75, 1.1, 0.9, 1.0, 0.75, 1.1,
                        0.9, 1.0, 0.75, 1.1, 0.9, 1.0, 0.75, 1.1],
        },
        'perc': {
            'timing': [0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2],
            'velocity': [1.0, 0.9, 1.0, 0.9, 1.0, 0.9, 1.0, 0.9,
                        1.0, 0.9, 1.0, 0.9, 1.0, 0.9, 1.0, 0.9],
        },
    },
)

GROOVE_ADVANCED_STRAIGHT = AdvancedGrooveProfile(
    "Straight", "No groove", {'perc': {'timing': [0] * 16, 'velocity': [1.0] * 16}}
)


class Pattern:
    """Represent a grid pattern with M instruments/samples over N steps."""

    def __init__(self, num_steps, instr_names, init=None):
        self.num_steps = num_steps
        self.num_instrs = len(instr_names)
        self.instr_names = instr_names
        if init is not None:
            self.array = np.array(init, dtype=np.float32)
        else:
            self.array = np.zeros((self.num_instrs, self.num_steps), dtype=np.float32)

    def randomize(self, p=0.5):
        on_off = np.random.rand(self.num_instrs, self.num_steps) < p
        velocities = np.random.uniform(0.5, 1.0, (self.num_instrs, self.num_steps))
        self.array = on_off * velocities

    def copy(self):
        return copy.deepcopy(self)

    def mutate(self, p=0.05):
        mut_toggle = np.random.rand(self.num_instrs, self.num_steps) < p
        new_velocities = np.random.uniform(0.5, 1.0, (self.num_instrs, self.num_steps))
        current_on_off = self.array > 0
        toggled_on_off = np.where(mut_toggle, np.logical_not(current_on_off), current_on_off)
        self.array = np.where(toggled_on_off,
                              np.where(current_on_off, self.array, new_velocities),
                              0)

    def mutate_samples(self, p=0.15):
        for idx in range(self.num_instrs):
            if np.random.uniform(0, 1) < p:
                self.instr_names[idx] = SampleManager.random(1)[0]

    def get(self, instr, step):
        return self.array[instr, step]

    def set(self, instr, step, val=1.0):
        self.array[instr, step] = val

    def render(self, num_repeats=4, bpm=120, sr=44100, pad_len=1,
               mix_vol=0.5, groove=GROOVE_ADVANCED_STRAIGHT):
        """Render the pattern applying groove and velocity-based decay."""
        samples = [SampleManager.load(name) for name in self.instr_names]
        instrument_categories = [categorize_sample(name) for name in self.instr_names]

        steps_per_beat = 4
        beat_len = 60 / bpm
        note_len = beat_len / steps_per_beat

        pat_len = num_repeats * self.num_steps * note_len + pad_len
        num_samples = int(pat_len * sr)
        audio = np.zeros(num_samples, dtype=np.float32)

        for step_idx in range(num_repeats * self.num_steps):
            current_step = step_idx % self.num_steps
            base_start_idx = int(step_idx * note_len * sr)

            for instr_idx in range(self.num_instrs):
                base_velocity = self.get(instr_idx, current_step)
                if base_velocity > 0:
                    category = instrument_categories[instr_idx]
                    gdata = groove.get_groove_for_instrument(category)

                    timing_offset_ms = gdata['timing'][current_step]
                    velocity_multiplier = gdata['velocity'][current_step]

                    start_idx = base_start_idx + int(timing_offset_ms / 1000 * sr)
                    final_velocity = np.clip(base_velocity * velocity_multiplier, 0.0, 1.0)

                    # Apply velocity-based decay before adjusting final volume.
                    base_sample = samples[instr_idx]
                    decayed_sample = apply_velocity_decay(base_sample, final_velocity, category)

                    sample_to_mix = decayed_sample * mix_vol * final_velocity
                    mix_sample(audio, sample_to_mix, start_idx)

        return audio


class SampleManager:
    """Find, load and cache samples."""

    cache = {}

    @classmethod
    def root_dir(cls):
        mod_path, _ = os.path.split(os.path.realpath(__file__))
        return os.path.realpath(os.path.join(mod_path, '..', 'samples'))

    @classmethod
    def get_list(cls, prefix=None):
        if not hasattr(cls, 'name_list'):
            root_path = cls.root_dir()
            names = []
            for file_root, _, files in os.walk(root_path, topdown=False):
                for name in files:
                    name, ext = name.split('.')
                    if ext != 'wav':
                        continue
                    relpath = os.path.relpath(file_root, root_path)
                    names.append(os.path.join(relpath, name))
            setattr(cls, 'name_list', names)
        names = getattr(cls, 'name_list')
        if prefix:
            names = [s for s in names if s.startswith(prefix)]
        return names

    @classmethod
    def get_path(cls, name):
        return os.path.join(cls.root_dir(), name + '.wav')

    @classmethod
    def load(cls, name):
        if name in cls.cache:
            return cls.cache[name]
        import soundfile as sf
        path = cls.get_path(name)
        data, sr = sf.read(path)
        assert sr == 44100
        if len(data.shape) == 2 and data.shape[1] == 2:
            data = 0.5 * (data[:, 0] + data[:, 1])
        cls.cache[name] = data.astype(np.float32)
        return cls.cache[name]

    @classmethod
    def random(cls, num_samples, prefix=None):
        import random
        return random.sample(cls.get_list(prefix), num_samples)


def mix_sample(audio, sample, start_idx):
    smp_len = sample.shape[0]
    end_idx = start_idx + smp_len
    if end_idx > audio.shape[-1]:
        end_idx = audio.shape[-1]
    if start_idx < 0:
        sample = sample[-start_idx:]
        start_idx = 0
    smp_len = min(sample.shape[0], end_idx - start_idx)
    audio[start_idx:end_idx] += sample[:smp_len]


##############################################################################


def new_pattern(fotf):
    samples = SampleManager.random(4)
    if fotf:
        samples[0] = SampleManager.random(1, 'drumhits/Kick')[0]
    return Pattern(16, samples)


def random_pattern(p=0.5, fotf=False, groove=GROOVE_ADVANCED_STRAIGHT):
    pat = new_pattern(fotf)
    pat.randomize(p)
    if fotf:
        for i in range(pat.num_steps):
            pat.set(0, i, 1.0 if i % 4 == 0 else 0.0)
    return pat, pat.render(groove=groove)


def mutate_pattern(pat, fotf=False):
    new_pat = pat.copy()
    new_pat.mutate()
    new_pat.mutate_samples()
    if fotf:
        for i in range(new_pat.num_steps):
            new_pat.set(0, i, 1.0 if i % 4 == 0 else 0.0)
        if new_pat.instr_names[0] != pat.instr_names[0]:
            new_pat.instr_names[0] = SampleManager.random(1, 'drumhits/Kick')[0]
    return new_pat


def hillclimb(model, fotf, target_score=1.0, target_dist=0.05, min_itrs=100,
              max_itrs=500, verbose=False, groove=GROOVE_ADVANCED_STRAIGHT):
    best_pattern, best_audio = random_pattern(fotf=fotf, groove=groove)
    best_p_good = model.eval_audio(best_audio)
    best_dist = abs(target_score - best_p_good)
    for i in range(max_itrs):
        new_pattern = mutate_pattern(best_pattern, fotf=fotf)
        audio = new_pattern.render(groove=groove)
        p_good = model.eval_audio(audio)
        dist = abs(target_score - p_good)
        if dist <= best_dist:
            if verbose:
                print(i, p_good)
            best_pattern, best_audio, best_p_good, best_dist = (
                new_pattern, audio, p_good, dist)
            if i >= min_itrs and best_dist <= target_dist:
                break
    return best_pattern, best_audio, best_p_good
