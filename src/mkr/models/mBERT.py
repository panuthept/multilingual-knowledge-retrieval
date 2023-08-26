from typing import List
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class mBERTReranker:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("amberoad/bert-multilingual-passage-reranking-msmarco")
        self.model = AutoModelForSequenceClassification.from_pretrained("amberoad/bert-multilingual-passage-reranking-msmarco")
        self.model.eval()

    def __call__(self, query: str, documents: List[str]):
        input_encoding = self.tokenizer([query] * len(documents), documents, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = self.model(**input_encoding).logits.detach().numpy()[:, 1]
        return outputs
    

if __name__ == "__main__":
    reranker = mBERTReranker()

    query = "ตาลีบันถูกโค่นล้มเมื่อไหร่"
    docs = [
        "กลุ่มก่อการร้ายอิสลามและกลุ่มการเมืองซึ่งปกครองพื้นที่ส่วนใหญ่ของประเทศอัฟกานิสถานและเมืองหลวงกรุงคาบูลในฐานะ \"รัฐอิสลามแห่งอัฟกานิสถาน\"",
        "ไม่ทราบราคาสินค้า",
        "ใช้สำหรับงานตกแต่งภายในทั่วไป กั้นห้อง หรือ แบ่งพื้นที่ภายในห้องออกจากกัน รวมทั้ง หน้าบานตู้ ประตู และหน้าต่าง",
        "11 กันยายน พ.ศ. 2544",
    ]

    print(reranker(query=query, documents=docs))