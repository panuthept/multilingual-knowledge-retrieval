from typing import List
from mkr.resources.resource_manager import ResourceManager
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from mkr.models.question_answering.extractive.baseclass import ExtractiveQA


class mRoBERTaExtractiveQA(ExtractiveQA):
    def __init__(self, model_name: str = "mRoBERTa"):
        self.resource_manager = ResourceManager()
        self.tokenizer = AutoTokenizer.from_pretrained(self.resource_manager.get_model_path(model_name, model_type="extractive_question_answering"))
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.resource_manager.get_model_path(model_name, model_type="extractive_question_answering"))
        self.model.eval()

    def predict_spans(self, queries: List[str], contexts: List[str]):
        if isinstance(queries, str):
            queries = [queries]
        if isinstance(contexts, str):
            contexts = [contexts]

        inputs = self.tokenizer(queries, contexts, padding=True, truncation=True, return_tensors="pt")
        outputs = self.model(**inputs)
        answer_start_scores = outputs.start_logits
        answer_end_scores = outputs.end_logits

        return answer_start_scores, answer_end_scores, inputs.input_ids
    

if __name__ == "__main__":
    extractiveQA = mRoBERTaExtractiveQA()

    query = "ตาลีบันถูกโค่นล้มเมื่อไหร่"
    docs = "กลุ่มก่อการร้ายอิสลามและกลุ่มการเมืองซึ่งปกครองพื้นที่ส่วนใหญ่ของประเทศอัฟกานิสถานและเมืองหลวงกรุงคาบูลในฐานะ \"รัฐอิสลามแห่งอัฟกานิสถาน\" ถูกก่อตั้งเมื่อ 11 กันยายน พ.ศ. 2524 และ ถูกโค้นล้มลงเมื่อ 11 กันยายน พ.ศ. 2544"

    answer_start_scores, answer_end_scores, token_ids = extractiveQA.predict_spans(query, docs)
    print(answer_start_scores)
    print(answer_end_scores)
    print(token_ids)