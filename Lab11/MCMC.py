import numpy as np
import collections
import os
import string


class TextMarkovChain:
    def __init__(self, to_learn: str):
        self.alphabet = self._create_alphabet()
        self.char_to_idx = {c: i for i, c in enumerate(self.alphabet)}
        self.vocab_size = len(self.alphabet)

        self.text = self._load_text_content(to_learn)
        self.encoded_text = self._encode_text(self.text)

        self.M = self._calculate_M()
        self.threshold = self._calculate_threshold()

    def _load_text_content(self, input_data: str) -> str:
        if os.path.exists(input_data) and os.path.isfile(input_data):
            try:
                with open(input_data, 'rb') as f:
                    raw = f.read()
                try:
                    return raw.decode('utf-8')
                except UnicodeDecodeError:
                    import chardet
                    enc = chardet.detect(raw)['encoding']
                    return raw.decode(enc, errors='ignore')
            except Exception:
                return input_data
        return input_data

    def _create_alphabet(self):
        alphabet = sorted(list(string.ascii_lowercase + " " + "абвгґдеєжзиіїйклмнопрстуфхцчшщьюя"))
        return alphabet

    def _encode_text(self, text: str) -> np.array:
        text = text.lower()
        indices = [self.char_to_idx[char] for char in text if char in self.char_to_idx]
        return np.array(indices, dtype=int)

    def _calculate_M(self):
        M = np.ones((self.vocab_size, self.vocab_size))
        if len(self.encoded_text) > 1:
            np.add.at(M, (self.encoded_text[:-1], self.encoded_text[1:]), 1)

        row_sums = M.sum(axis=1, keepdims=True)
        probs = M / row_sums
        return np.log(probs)

    def _calculate_threshold(self, chunks: int = 100, chunk_len: int = 100):
        scores = []
        starts = np.random.randint(0, len(self.encoded_text) - chunk_len, chunks)
        for start in starts:
            chunk = self.encoded_text[start : start + chunk_len]
            score = self.M[chunk[:-1], chunk[1:]]
            scores.append(np.mean(score))
        
        mean = np.mean(scores)
        std = np.std(scores)
        
        threshold = mean - (3 * std)

        return threshold

    def score(self, to_score: str, check=False):
        text_content = self._load_text_content(to_score)
        indices = self._encode_text(text_content)

        score = np.mean(self.M[indices[:-1], indices[1:]])
        return score, score > self.threshold if check else score


if __name__ == '__main__':
    model = TextMarkovChain("./Lab11/Data/TheWarOfTheWorlds.txt")
    res, status = model.score("./Lab11/Data/TheTimeMachine.txt", check=True)
    res_bad, status_bad = model.score("./Lab11/Data/TheBadText.txt", check=True)
    print(f"[BIG DATA] Threshold: {model.threshold:.3f}")
    print(f"[Score Good] {res:.2f}, [Pass] {status} | [Score Bad] {res_bad:.2f}, [Pass] {status_bad}")

    model = TextMarkovChain("./Lab11/Data/nestayko-vsevolod-zinoviyovych-toreadory-z-vasiukivky914.txt")
    res, status = model.score("Як тебе звати? Привіт Марк, я тест.", check=True)
    res_bad, status_bad = model.score("піцк іацк аіа піуі.", check=True)
    print(f"[SMALL DATA] Threshold: {model.threshold:.3f}")
    print(f"[Score Good] {res:.2f}, [Pass] {status} | [Score Bad] {res_bad:.2f}, [Pass] {status_bad}")
