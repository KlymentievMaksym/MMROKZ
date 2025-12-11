import random
import numpy as np
import collections
import chardet
import string
from tqdm import trange


class TextMarkovChain:
    def __init__(self, to_learn: str):
        try:
            self.text = self._load_text(to_learn)
        except Exception as e:
            self.text = to_learn
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
        # alphabet = list(set(self.text))
        # alphabet.remove(" ")
        alphabet = sorted(list(string.ascii_lowercase + " " + "абвгґдеєжзиіїйклмнопрстуфхцчшщьюя"))
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

    def score(self, to_score: str):
        try:
            raw_score = self._load_text(to_score)
        except Exception as e:
            raw_score = to_score
        cleaned_score = self._clean(raw_score)

        score = 0.0
        for i in range(len(cleaned_score) - 1):
            char_first = cleaned_score[i].lower()
            char_second = cleaned_score[i+1].lower()
            if char_first in self.alphabet and char_second in self.alphabet:
                score += self.M[self.char_to_idx[char_first], self.char_to_idx[char_second]]

        return score/len(cleaned_score) if len(cleaned_score) > 0 else score


if __name__ == '__main__':
    model = TextMarkovChain("./Lab11/Data/TheWarOfTheWorlds.txt")
    res = model.score("./Lab11/Data/TheTimeMachine.txt")
    res_bad = model.score("./Lab11/Data/TheBadText.txt")
    print("[BIG DATA]")
    print(f"[Score Good] {res:.2f}, [Score Bad] {res_bad:.2f}")

    model = TextMarkovChain("./Lab11/Data/nestayko-vsevolod-zinoviyovych-toreadory-z-vasiukivky914.txt")
    res = model.score("Як тебе звати? Привіт Марк, я тест.")
    res_bad = model.score("піцк іацк аіа піуі.")
    print("[SMALL DATA]")
    print(f"[Score Good] {res:.2f}, [Score Bad] {res_bad:.2f}")
