import os
import math
import json
import random
import argparse
import numpy as np
import tensorflow_text
from tqdm import trange
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.utils import Sequence, SequenceEnqueuer


def cal_mrr(targets, preds, *args, **kwargs):
    targets = tf.argmax(targets, axis=-1)   # Convert one-hot to index
    targets = tf.reshape(targets, [-1, 1])
    sorted_indices = tf.argsort(preds, axis=-1, direction="DESCENDING")

    targets = tf.cast(targets, sorted_indices.dtype)

    matched_indices = tf.where(sorted_indices == targets)[:, -1]
    mrr = tf.reduce_mean(1 / (matched_indices + 1))
    return mrr


class DPRDataset(Sequence):
    def __init__(
            self, 
            corpus_path: str, 
            qrels_path: str, 
            batch_size: int = 32, 
            shuffle: bool = True,
            negative_size: int = 10,
            hard_negative_path: str = None,
            hard_negative_factor: float = 2.0,
        ):
        self.corpus_path = corpus_path
        self.qrels_path = qrels_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.negative_size = negative_size
        self.hard_negative_path = hard_negative_path
        self.hard_negative_factor = hard_negative_factor

        self.corpus, self.qrels = self._read_data()
        self.unique_doc_ids = set(self.corpus.values())

        self.hard_negative_ids = {}
        if self.hard_negative_path is not None:
            with open(self.hard_negative_path, "r") as f:
                for line in f:
                    data = json.loads(line.strip())
                    question = data["question"]
                    hard_negative_ids = data["top1000"]
                    self.hard_negative_ids[question] = hard_negative_ids[:min(int(self.negative_size * self.hard_negative_factor), len(hard_negative_ids))]

        self.num_batches = len(self.qrels) // self.batch_size
        self.indices = list(range(len(self.qrels)))
        if shuffle:
            np.random.shuffle(self.indices)

    def _read_data(self):
        # Read corpus
        corpus = {}
        with open(self.corpus_path, "r") as f:
            for line in f:
                data = json.loads(line.strip())
                hash = data["hash"]
                content = data["content"]
                metadata = data["metadata"]
                title = metadata["title"]
                corpus[hash] = f"{title}\n{content}"
        # Read qrels
        qrels = []
        with open(self.qrels_path, "r") as f:
            for line in f:
                qrel = json.loads(line.strip())
                question = qrel["question"]
                if question == "":
                    continue
                context_answers = qrel["context_answers"]
                if len(context_answers) == 0:
                    continue
                hashs = list(context_answers.keys())
                qrels.append([question, hashs])
        return corpus, qrels

    def collate_fn(self, questions, documents, labels):
        # Cast type to tensor
        questions = tf.convert_to_tensor(questions, dtype=tf.string)
        documents = tf.convert_to_tensor(documents, dtype=tf.string)
        labels = tf.convert_to_tensor(labels, dtype=tf.int32)
        return (questions, documents), labels

    def __len__(self):
        return len(self.qrels)

    def __getitem__(self, idx):
        indices = self.indices[idx * self.batch_size: (idx + 1) * self.batch_size]
        qrels = [self.qrels[i] for i in indices]
        questions, hashss = zip(*qrels)
        positive_doc_ids = [random.choice(hashs) for hashs in hashss]

        cand_docs = []
        for question, positive_doc_id in zip(questions, positive_doc_ids):
            negative_docs = []
            if self.negative_size > 0:
                if len(self.hard_negative_ids) > 0:
                    negative_doc_ids = list(set(self.hard_negative_ids[question]) - set([positive_doc_id]))
                else:
                    negative_doc_ids = list(self.unique_doc_ids - set([positive_doc_id]))
                negative_doc_ids = np.random.choice(negative_doc_ids, size=self.negative_size, replace=False)
                negative_docs.extend([self.corpus[doc_id] for doc_id in negative_doc_ids])
            positive_doc = self.corpus[positive_doc_id]

            cand_docs.append([positive_doc] + negative_docs)
        labels = np.zeros((len(questions), self.negative_size + 1), dtype=np.int32)
        labels[:, 0] = 1
        return self.collate_fn(questions, cand_docs, labels)

    def on_epoch_end(self):
        np.random.shuffle(self.indices)


def DPRTrainerTF(model, batch_size: int = 32, negative_size: int = 10):
    queries = Input(
        shape=[],
        batch_size=batch_size,
        dtype=tf.string
    )
    documents = Input(
        shape=[negative_size + 1],
        batch_size=batch_size,
        dtype=tf.string
    )

    flatten_documents = tf.reshape(documents, [-1])

    encoded_queries = model(queries)
    encoded_documents = model(flatten_documents)

    # NOTE: We don't use in-batch negative because there are many questions that have the same positive document.
    encoded_queries = tf.reshape(encoded_queries, [batch_size, 1, -1])
    encoded_documents = tf.reshape(encoded_documents, [batch_size, negative_size + 1, -1])

    similarity_scores = tf.matmul(encoded_queries, encoded_documents, transpose_b=True)
    similarity_scores = tf.reshape(similarity_scores, [batch_size, negative_size + 1])
    return Model(inputs=[queries, documents], outputs=similarity_scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--learning_rate_decay", type=float, default=0.01)
    parser.add_argument("--model_path", type=str, default="./models/universal-sentence-encoder-multilingual_3")
    parser.add_argument("--dataset_path", type=str, default="corpus/training_dataset")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--save_dir", type=str, default="finetuned_models")
    parser.add_argument("--save_name", type=str, default="mUSE")
    parser.add_argument("--negative_size", type=int, default=10)
    parser.add_argument("--hard_negative_factor", type=float, default=2.0)
    parser.add_argument("--validation_step", type=int, default=10)
    args = parser.parse_args()

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    training_dataset = DPRDataset(
        corpus_path=os.path.join(args.dataset_path, "corpus.jsonl"), 
        qrels_path=os.path.join(args.dataset_path, "qrel_train.jsonl"),
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        negative_size=args.negative_size,
        hard_negative_path=os.path.join(args.dataset_path, "bm25_top1000.jsonl"),
        hard_negative_factor=args.hard_negative_factor,
    )
    print(f"Training size: {len(training_dataset)}")
    if os.path.exists(os.path.join(args.dataset_path, "qrel_val.jsonl")):
        validation_dataset = DPRDataset(
            corpus_path=os.path.join(args.dataset_path, "corpus.jsonl"), 
            qrels_path=os.path.join(args.dataset_path, "qrel_val.jsonl"),
            batch_size=args.batch_size,
            shuffle=False,
            negative_size=args.negative_size,
            hard_negative_path=os.path.join(args.dataset_path, "bm25_top1000.jsonl"),
            hard_negative_factor=args.hard_negative_factor,
        )
        print(f"Validation size: {len(validation_dataset)}")

    hub_load = hub.load(args.model_path)
    muse = hub.KerasLayer(
        hub_load,
        input_shape=(),
        output_shape=(512, ),
        dtype=tf.string,
        trainable=True
    )
    trainer = DPRTrainerTF(muse, args.batch_size, args.negative_size)
    trainer.summary()
    trainer.compile(optimizer=AdamW(learning_rate=args.learning_rate), loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, axis=-1), metrics=cal_mrr)

    train_summary_writer = tf.summary.create_file_writer(args.log_dir)

    step = 0
    best_score = -math.inf
    for epoch_idx in trange(args.epochs, desc="Epoch"):
        for train_batch_idx in trange(training_dataset.num_batches, desc="Training step"):
            step += 1
            # Training step
            (queries, documents), labels = training_dataset[train_batch_idx]
            train_outputs = trainer.train_on_batch(
                x=[queries, documents],
                y=labels,
                return_dict=True,
            )
            train_loss = train_outputs["loss"]
            train_mrr = train_outputs["cal_mrr"]
            with train_summary_writer.as_default():
                tf.summary.scalar("Train_Loss", train_loss, step)
                tf.summary.scalar("Train_MRR", train_mrr, step)
                tf.summary.scalar("Learning_Rate", K.get_value(trainer.optimizer.learning_rate), step)

            # Validation step
            if step % args.validation_step == 0:
                if "validation_dataset" in locals():
                    val_losses = []
                    val_metrics = []
                    for val_batch_idx in range(validation_dataset.num_batches):
                        (queries, documents), labels = validation_dataset[val_batch_idx]
                        val_outputs = trainer.test_on_batch(
                            x=[queries, documents],
                            y=labels,
                            return_dict=True,
                        )
                        val_loss = val_outputs["loss"]
                        val_mrr = val_outputs["cal_mrr"]
                        val_losses.append(val_loss)
                        val_metrics.append(val_mrr)
                    val_losses = np.mean(val_losses)
                    val_metrics = np.mean(val_metrics)
                    with train_summary_writer.as_default():
                        tf.summary.scalar("Valiation_Loss", val_losses, step)
                        tf.summary.scalar("Valiation_MRR", val_metrics, step)
                    if val_metrics > best_score:
                        best_score = val_metrics
                        tf.saved_model.save(hub_load, os.path.join(args.save_dir, args.save_name, "best_model"))
                        with open(os.path.join(args.save_dir, args.save_name, "best_scores.txt"), "a") as f:
                            f.write(f"{best_score} - {epoch_idx}\n")
                else:
                    tf.saved_model.save(hub_load, os.path.join(args.save_dir, args.save_name, "best_model"))
        training_dataset.on_epoch_end()
        K.set_value(trainer.optimizer.learning_rate, K.get_value(trainer.optimizer.learning_rate) * (1 - args.learning_rate_decay))

    # Validation step
    if "validation_dataset" in locals():
        val_losses = []
        val_metrics = []
        for val_batch_idx in range(validation_dataset.num_batches):
            (queries, documents), labels = validation_dataset[val_batch_idx]
            val_outputs = trainer.test_on_batch(
                x=[queries, documents],
                y=labels,
                return_dict=True,
            )
            val_loss = val_outputs["loss"]
            val_mrr = val_outputs["cal_mrr"]
            val_losses.append(val_loss)
            val_metrics.append(val_mrr)
        val_losses = np.mean(val_losses)
        val_metrics = np.mean(val_metrics)
        with train_summary_writer.as_default():
            tf.summary.scalar("Valiation_Loss", val_losses, step)
            tf.summary.scalar("Valiation_MRR", val_metrics, step)
        if val_metrics > best_score:
            best_score = val_metrics
            tf.saved_model.save(hub_load, os.path.join(args.save_dir, args.save_name, "best_model"))
            with open(os.path.join(args.save_dir, args.save_name, "best_scores.txt"), "a") as f:
                f.write(f"{best_score}\n")
    else:
        tf.saved_model.save(hub_load, os.path.join(args.save_dir, args.save_name, "best_model"))