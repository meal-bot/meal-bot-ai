from kiwipiepy import Kiwi

ALLOWED_POS = {"NNG", "NNP", "VA", "VV", "SL", "SN"}
MIN_TOKEN_LEN = 2


class KiwiTokenizer:
    def __init__(self):
        self.kiwi = Kiwi()

    def tokenize(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        tokens = []
        for token in self.kiwi.tokenize(text):
            if token.tag not in ALLOWED_POS:
                continue
            form = token.form.lower().strip()
            if not form:
                continue
            # 숫자(SN)는 1자도 허용, 나머지는 2자 이상
            if token.tag != "SN" and len(form) < MIN_TOKEN_LEN:
                continue
            tokens.append(form)
        return tokens