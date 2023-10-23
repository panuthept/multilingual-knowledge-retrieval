from dataclasses import dataclass
from mkr.models.question_answering.extractive.mRoBERTa import ExtractiveQA, mRoBERTaExtractiveQA


@dataclass
class AnswerExtractorConfig:
    model_name: str


class AnswerExtractor:
    available_models = ["mRoBERTa"]

    def __init__(self, config: AnswerExtractorConfig):
        self.model_name = config.model_name
        self.extractor: ExtractiveQA = self._load_extractor(self.model_name)

    def _load_extractor(self, model_name: str) -> ExtractiveQA:
        # Load extractor
        assert model_name in self.available_models, f"Unknown extractor: {model_name}"
        if "mRoBERTa" in model_name:
            extractor = mRoBERTaExtractiveQA(model_name)
        else:
            raise ValueError(f"Unknown extractor: {model_name}")
        return extractor
    
    def __call__(
            self,
            query: str,
            context: str,
    ):
        answer_start_scores, answer_end_scores, token_ids = self.extractor.predict_spans(query, context)

        answer_start_index = answer_start_scores[0].argmax()
        answer_end_index = answer_end_scores[0].argmax() + 1
        answer_tokens = token_ids[0][answer_start_index:answer_end_index]

        answer = self.extractor.tokenizer.decode(answer_tokens)
        return answer
    

if __name__ == "__main__":
    extractiveQA = AnswerExtractor(AnswerExtractorConfig(model_name="mRoBERTa"))

    query = "ตาลีบันถูกโค่นล้มเมื่อไหร่"
    doc = "กลุ่มก่อการร้ายอิสลามและกลุ่มการเมืองซึ่งปกครองพื้นที่ส่วนใหญ่ของประเทศอัฟกานิสถานและเมืองหลวงกรุงคาบูลในฐานะ \"รัฐอิสลามแห่งอัฟกานิสถาน\" ถูกก่อตั้งเมื่อ 11 กันยายน พ.ศ. 2524 และ ถูกโค้นล้มลงเมื่อ 11 กันยายน พ.ศ. 2544"

    answer = extractiveQA(query, doc)
    print(answer)