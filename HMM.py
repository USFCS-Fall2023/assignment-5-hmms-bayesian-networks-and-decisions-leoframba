import random
import argparse
import codecs
import os
import numpy as np


# observations
class Observation:
    def __init__(self, stateseq, outputseq):
        self.stateseq = stateseq  # sequence of states
        self.outputseq = outputseq  # sequence of outputs

    def __str__(self):
        return ' '.join(self.stateseq) + '\n ' + ' '.join(self.outputseq) + '\n'

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.outputseq)


# hmm model
class HMM:
    def __init__(self, transitions={}, emissions={}):
        """creates a model from transition and emission probabilities"""
        ## Both of these are dictionaries of dictionaries. e.g. :
        # {'#': {'C': 0.814506898514, 'V': 0.185493101486},
        #  'C': {'C': 0.625840873591, 'V': 0.374159126409},
        #  'V': {'C': 0.603126993184, 'V': 0.396873006816}}

        self.transitions = transitions
        self.emissions = emissions

    ## part 1 - you do this.
    def load(self, basename):
        """reads HMM structure from transition (basename.trans),
        and emission (basename.emit) files,
        as well as the probabilities."""

        with open(basename + '.trans', 'r') as file:
            for line in file:
                split_line = line.split()
                if split_line[0] not in self.transitions:
                    self.transitions[split_line[0]] = {}
                self.transitions[split_line[0]][split_line[1]] = float(split_line[2])

        with open(basename + '.emit', 'r') as file:
            for line in file:
                split_line = line.split()
                if split_line[0] not in self.emissions:
                    self.emissions[split_line[0]] = {}
                self.emissions[split_line[0]][split_line[1]] = float(split_line[2])

    ## you do this.
    def generate(self, n):
        """return an n-length observation by randomly sampling from this HMM."""
        # Start with the initial state
        current_state = '#'
        stateseq = []
        outputseq = []

        for i in range(n):
            state_dict = self.transitions[current_state]
            current_state = random.choices(list(state_dict.keys()), weights=list(state_dict.values()), k=1)[0]
            stateseq.append(current_state)
            emit_dict = self.emissions[current_state]
            emission = random.choices(list(emit_dict.keys()), weights=list(emit_dict.values()), k=1)[0]
            outputseq.append(emission)

        return Observation(stateseq, outputseq)

    ## you do this: Implement the Viterbi alborithm. Given an Observation (a list of outputs or emissions)
    ## determine the most likely sequence of states.

    def viterbi(self, observation):
        states = list(self.transitions.keys())
        states.remove('#')
        T = len(observation.outputseq)
        S = len(states)
        V = np.zeros((S, T))
        path = np.zeros((S, T), dtype=int)

        # base
        for s in range(S):
            V[s, 0] = self.transitions['#'].get(states[s], 0) * self.emissions[states[s]].get(observation.outputseq[0],
                                                                                              0)

        for t in range(1, T):
            for s in range(S):
                # Compute the max probability for each state
                (prob, state) = max(
                    (V[s_prime, t - 1] * self.transitions[states[s_prime]].get(states[s], 0) * self.emissions[
                        states[s]].get(observation.outputseq[t], 0), s_prime)
                    for s_prime in range(S)
                )
                V[s, t] = prob
                path[s, t] = state

        last_state = np.argmax(V[:, T - 1])
        best_path = [last_state]

        for t in range(T - 1, 0, -1):
            last_state = path[last_state, t]
            best_path.insert(0, last_state)

        best_path_states = [states[i] for i in best_path]
        print(f"Viterbi for \"{' '.join(observation.outputseq)}\": {' '.join(best_path_states)}")

    def parse_file_to_observations(self, observation_file, algorithm='forward'):
        observations = []
        with open(observation_file, 'r') as file:
            prev_line = None
            for line in file:
                if prev_line is None:
                    prev_line = line
                    continue
                if algorithm == 'forward':
                    self.forward(Observation(prev_line.split(), line.split()))
                elif algorithm == 'viterbi':
                    self.viterbi(Observation(prev_line.split(), line.split()))
                observations.append(Observation(prev_line.split(), line.split()))
                prev_line = None

        return observations

    def forward(self, observation):
        def state_index(state):
            """Return the index of the given state."""
            return list(self.transitions).index(state)

        obs_len = len(observation)
        state_len = len(self.transitions)

        M = np.zeros((state_len, obs_len))
        M[state_index('#'), 0] = 1

        for state in self.transitions.keys():
            if state == '#':
                continue
            if self.emissions[state].get(observation.outputseq[0]) is not None:
                M[state_index(state), 1] = self.transitions['#'][state] * self.emissions[state][
                    observation.outputseq[0]]

        for i in range(2, obs_len):
            for state in self.transitions.keys():
                if state == '#':
                    continue
                if self.emissions[state].get(observation.outputseq[i - 1]) is not None:
                    M[state_index(state), i] = self.emissions[state][observation.outputseq[i - 1]] * sum(
                        [M[state_index(prev_state), i - 1] * self.transitions[prev_state][state] for prev_state in
                         self.transitions.keys() if prev_state != '#'])

        max_index = np.argmax(M[:, obs_len - 1])
        next = list(self.transitions.keys())[max_index]
        print(f"Forward for \"{' '.join(observation.outputseq)}\": {next}")


def is_empty_or_whitespace(s):
    return not s or s.isspace()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HMM trainer')
    parser.add_argument('basename', type=str,
                        help='basename for training files')
    parser.add_argument('--generate', type=int,
                        help='Generate n observations')

    parser.add_argument('--forward', type=str,
                        help='Run forward algorithm')

    parser.add_argument('--viterbi', type=str,
                        help='Run viterbi algorithm')
    args = parser.parse_args()

    model = HMM()
    model.load(args.basename)

    if args.generate:
        print(model.generate(args.generate))

    if args.forward:
        model.parse_file_to_observations(args.forward, 'forward')

    if args.viterbi:
        model.parse_file_to_observations(args.viterbi, 'viterbi')
