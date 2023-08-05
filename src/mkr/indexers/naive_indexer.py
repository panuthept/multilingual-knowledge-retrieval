import json
import argparse
import numpy as np

from mkr.encoders.mUSE import mUSESentenceEncoder


class NaiveIndexer:
    def __init__(self, encoder_name: str):
        # Load encoder
        if encoder_name == "mUSE":
            self.encoder = mUSESentenceEncoder()
        else:
            raise ValueError(f"Unknown encoder: {encoder_name}")
        
    def __call__(self, input_file: str, output_file: str, batch_size: int = 32):
        # Encode sentences in the input file (jsonl format)
        doc_texts = []
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                doc_texts.append(data["doc_text"])
        doc_embs = self.encoder.encode_batch(doc_texts, batch_size=batch_size)

        # Save encoded sentences (numpy format)
        np.save(output_file, doc_embs.numpy())


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=str, help="Input file with sentences to encode")
    parser.add_argument("--output", required=True, type=str, help="Output file with encoded sentences")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for encoding")
    parser.add_argument("--encoder", default="mUSE", type=str, help="Encoder to use")
    args = parser.parse_args()

    # Load encoder
    if args.encoder == "mUSE":
        encoder = mUSESentenceEncoder()
    else:
        raise ValueError(f"Unknown encoder: {args.encoder}")

    # Encode sentences in the input file (jsonl format)
    doc_texts = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            doc_texts.append(data["doc_text"])
    doc_embs = encoder.encode_batch(doc_texts, batch_size=args.batch_size)

    # Save encoded sentences (numpy format)
    np.save(args.output, doc_embs.numpy())