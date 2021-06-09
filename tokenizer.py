import re

from vncorenlp import VnCoreNLP


class VnCoreTokenizer:
    def __init__(self, path="vncorenlp/VnCoreNLP-1.1.1.jar"):
        self.rdrsegmenter = VnCoreNLP(path, annotators="wseg", max_heap_size="-Xmx500m")

    def tokenize(self, text: str, return_sentences=False) -> str:
        sentences = self.rdrsegmenter.tokenize(text)
        if return_sentences:
            return [" ".join(sentence) for sentence in sentences]
        output = ""

        for sentence in sentences:
            output += " ".join(sentence) + " "
        return self._strip_white_space(output)

    def _strip_white_space(self, text):
        text = re.sub("\n+", "\n", text).strip()
        text = re.sub(" +", " ", text).strip()
        return text


# if __name__ == "__main__":
#     tokenizer = VnCoreTokenizer("./vncorenlp/VnCoreNLP-1.1.1.jar")
#     print(
#         tokenizer.tokenize(
#             "Tôi là Nguyễn Đức Long sinh viên đại học Bách Khoa Hà Nội.
#               Lớp Công nghệ thông tin"
#         )
#     )
