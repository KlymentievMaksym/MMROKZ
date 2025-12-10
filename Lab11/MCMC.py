import random
import numpy as np
import collections
import chardet
import string
from tqdm import trange

alphabet = list(string.ascii_lowercase)
alphabet_set = set(alphabet)
epsilon = 1e-12

def make_cipher_map():
    shuffled = alphabet.copy()
    random.shuffle(shuffled)
    return dict(zip(alphabet, shuffled))

def cipher(text, cipher_map: dict = None):
    if cipher_map is None:
        cipher_map = make_cipher_map()
    result = []
    for ch in text:
        low = ch.lower()
        if low in cipher_map:
            mapped = cipher_map[low]
            # preserve case
            result.append(mapped.upper() if ch.isupper() else mapped)
        else:
            result.append(ch)
    return ''.join(result)

def clean(text):
    return ''.join(ch for ch in text.lower() if ch in alphabet_set)

def load_text(path: str) -> str:
    with open(path, 'rb') as f:
        raw = f.read()
    enc = chardet.detect(raw)['encoding'] or 'utf-8'
    return raw.decode(enc, errors='ignore')

def get_bigrams(text):
    return [text[i:i+2] for i in range(len(text)-1)]

def receive_array_counts(counter: collections.Counter, learn_unique_character: list, smoothing: float = 1.0):
    len_ch = len(learn_unique_character)
    indexes = {char: i for i, char in enumerate(learn_unique_character)}
    counts = np.zeros((len_ch, len_ch), dtype=float)

    for bigram, cnt in counter.items():
        a, b = bigram[0], bigram[1]
        if a in indexes and b in indexes:
            counts[indexes[a], indexes[b]] = cnt

    counts += smoothing
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    probs = counts / row_sums
    return probs

def log_score(mapping: dict, logM: np.array, ciphered_text: str, learn_indexes: dict):
    score = 0.0
    for i in range(len(ciphered_text) - 1):
        char_first = ciphered_text[i].lower()
        char_second = ciphered_text[i+1].lower()
        if char_first in mapping and char_second in mapping:
            char_first_mapped = mapping[char_first]
            char_second_mapped = mapping[char_second]
            if char_first_mapped in learn_indexes and char_second_mapped in learn_indexes:
                score += logM[learn_indexes[char_first_mapped], learn_indexes[char_second_mapped]]
            else:
                score += np.log(epsilon)
        else:
            score += np.log(epsilon)
    return score

def swap_mapping(mapping: dict):
    new_map = mapping.copy()
    a, b = random.sample(alphabet, 2)
    new_map[a], new_map[b] = new_map[b], new_map[a]
    return new_map

def find_best_score(f: dict, logM: np.array, ciphered_text: str, learn_indexes: dict, current_score: float):
    f_candidate = swap_mapping(f)
    score_candidate = log_score(f_candidate, logM, ciphered_text, learn_indexes)

    if score_candidate >= current_score:
        return f_candidate, score_candidate
    else:
        u = random.random()
        prob = np.exp(score_candidate - current_score)
        if u < prob:
            return f_candidate, score_candidate
        else:
            return f, current_score

def mcmc_text(path_to_learn: str, path_to_score: str, iterations: int = 2000, report_every: int = 1000, do_print: bool = False):
    raw_learn = load_text(path_to_learn)
    raw_score = load_text(path_to_score)

    text_to_learn = clean(raw_learn)
    cleaned_score = clean(raw_score)

    learn_bigrams = get_bigrams(text_to_learn)
    counter = collections.Counter(learn_bigrams)
    learn_unique_character = sorted(list(set(text_to_learn)))
    learn_indexes = {c: i for i, c in enumerate(learn_unique_character)}

    M = receive_array_counts(counter, learn_unique_character, smoothing=1.0)
    logM = np.log(M + epsilon)

    f = dict(zip(alphabet, alphabet))
    current_score = log_score(f, logM, cleaned_score, learn_indexes)
    best_map = f.copy()
    best_score = current_score

    if do_print:
        print(f"Start score: {current_score:.2f}, unique learn chars: {len(learn_unique_character)}\n")
    for it in trange(1, iterations + 1):
        f, current_score = find_best_score(f, logM, cleaned_score, learn_indexes, current_score)
        if current_score > best_score:
            best_score = current_score
            best_map = f.copy()
        if do_print and it % report_every == 0 or it == 1:
            print(f"\niter {it}/{iterations} | curr {current_score:.2f} | best {best_score:.2f}\n")

    full_score = []
    for char in raw_score:
        char_low = char.lower()
        if char_low in best_map:
            mapped = best_map[char_low]
            full_score.append(mapped.upper() if char.isupper() else mapped)
        else:
            full_score.append(char)
    full_text_score = ''.join(full_score)

    if do_print:
        print("\n[Example outputs]\n")
        index = random.randint(0, len(raw_score) - 200)
        print("[Ciphered sample]\n", raw_score[index:index+200])
        print("\n[Deciphered sample]\n", full_text_score[index:index+200])
    return {
        "best_map": best_map,
        "best_score": best_score,
        "deciphered_text": full_text_score,
    }

class TextMarkovChain:
    def __init__(self, path_to_learn: str):
        self.text = self._load_text(path_to_learn)
        self.alphabet = self._create_alphabet()
        self.text_to_learn = self._clean(self.text)
        self.char_to_idx = {c: i for i, c in enumerate(self.alphabet)}
        self.vocab_size = len(self.alphabet)
        self.M = self._calculate_M()

    def _load_text(self, path: str) -> str:
        with open(path, 'rb') as f:
            raw = f.read()
        enc = chardet.detect(raw)['encoding'] or 'utf-8'
        return raw.decode(enc, errors='ignore')

    def _clean(self, text):
        return ''.join(ch for ch in text.lower() if ch in self.alphabet)

    def _get_bigrams(self, text):
        return [text[i:i+2] for i in range(len(text)-1)]

    def _create_alphabet(self):
        alphabet = list(set(self.text))
        alphabet.remove(" ")
        return alphabet

    def _calculate_M(self):
        M = np.ones((self.vocab_size, self.vocab_size))
        counter = collections.Counter(self._get_bigrams(self.text_to_learn))
        for bigram, cnt in counter.items():
            a, b = bigram[0], bigram[1]
            if a in self.alphabet and b in self.alphabet:
                M[self.char_to_idx[a], self.char_to_idx[b]] += cnt

        row_sums = M.sum(axis=1, keepdims=True)
        probs = M / row_sums
        return np.log(probs)

    def score(self, path_to_score: str):
        raw_score = self._load_text(path_to_score)
        cleaned_score = self._clean(raw_score)
        score = 0.0
        for i in range(len(cleaned_score) - 1):
            char_first = cleaned_score[i].lower()
            char_second = cleaned_score[i+1].lower()
            if char_first in self.alphabet and char_second in self.alphabet:
                score += self.M[self.char_to_idx[char_first], self.char_to_idx[char_second]]

        return score


if __name__ == '__main__':
    # res = mcmc_text("./Data/TheWarOfTheWorlds.txt", "./Data/TheTimeMachine.txt", iterations=1500, report_every=100)
    res = TextMarkovChain("./Lab11/Data/TheWarOfTheWorlds.txt").score("./Lab11/Data/TheTimeMachine.txt")
    print(f"Score: {res:.2f}")
