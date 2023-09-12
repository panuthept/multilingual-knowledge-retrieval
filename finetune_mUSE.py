import os
import math
import json
import argparse
import numpy as np
import tensorflow_text
from tqdm import trange
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Adam


class DPRDataset(Sequence):
    def __init__(
            self, 
            corpus_path: str, 
            qrels_path: str, 
            batch_size: int = 32, 
            shuffle: bool = True,
            negative_mining: str = "in-batch",
        ):
        self.corpus_path = corpus_path
        self.qrels_path = qrels_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.negative_mining = negative_mining

        self.data = self._read_data()

        self.num_batches = len(self.data) // self.batch_size
        self.indices = list(range(len(self.data)))
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
        data = []
        with open(self.qrels_path, "r") as f:
            for line in f:
                qrel = json.loads(line.strip())
                question = qrel["question"]
                context_answers = qrel["context_answers"]
                if len(context_answers) == 0:
                    continue
                hashs = list(context_answers.keys())
                for hash in hashs:
                    data.append([question, corpus[hash]])
        return data

    def collate_fn(self, batch_data):
        if self.negative_mining == "in-batch":
            queries, documents = zip(*batch_data)
            labels = np.diagflat(np.ones(len(queries), dtype=np.int32))
            # Cast type to tensor
            queries = tf.convert_to_tensor(queries, dtype=tf.string)
            documents = tf.convert_to_tensor(documents, dtype=tf.string)
            labels = tf.convert_to_tensor(labels, dtype=tf.int32)
            return (queries, documents), labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_data = [self.data[i] for i in batch_indices]
        return self.collate_fn(batch_data)

    def on_epoch_end(self):
        np.random.shuffle(self.indices)


def DPRTrainerTF(model):
    queries = Input(
        shape=[],
        dtype=tf.string
    )
    documents = Input(
        shape=[],
        dtype=tf.string
    )

    encoded_queries = model(queries)
    encoded_documents = model(documents)

    similarity_scores = tf.matmul(encoded_queries, encoded_documents, transpose_b=True)
    return Model(inputs=[queries, documents], outputs=similarity_scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--learning_rate_decay", type=float, default=0.01)
    parser.add_argument("--model_path", type=str, default="./models/universal-sentence-encoder-multilingual_3")
    parser.add_argument("--dataset_path", type=str, default="corpus/training_dataset")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--save_dir", type=str, default="finetuned_models")
    parser.add_argument("--save_name", type=str, default="mUSE")
    parser.add_argument("--validation_step", type=int, default=10)
    args = parser.parse_args()

    training_dataset = DPRDataset(
        corpus_path=os.path.join(args.dataset_path, "corpus.jsonl"), 
        qrels_path=os.path.join(args.dataset_path, "qrel_train.jsonl"),
        batch_size=args.batch_size,
        shuffle=args.shuffle,
    )
    print(f"Training size: {len(training_dataset)}")
    if os.path.exists(os.path.join(args.dataset_path, "qrel_val.jsonl")):
        validation_dataset = DPRDataset(
            corpus_path=os.path.join(args.dataset_path, "corpus.jsonl"), 
            qrels_path=os.path.join(args.dataset_path, "qrel_val.jsonl"),
            batch_size=args.batch_size,
            shuffle=False,
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
    trainer = DPRTrainerTF(muse)
    trainer.compile(optimizer=Adam(learning_rate=args.learning_rate), loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, axis=-1))

    train_summary_writer = tf.summary.create_file_writer(args.log_dir)

    step = 0
    best_loss = math.inf
    for epoch_idx in trange(args.epochs, desc="Epoch"):
        for train_batch_idx in trange(training_dataset.num_batches, desc="Training step"):
            step += 1
            # Training step
            (queries, documents), labels = training_dataset[train_batch_idx]
            train_loss = trainer.train_on_batch(
                x=[queries, documents],
                y=labels,
            )
            with train_summary_writer.as_default():
                tf.summary.scalar("Train_Loss", train_loss, step)
                tf.summary.scalar("Learning_Rate", K.get_value(trainer.optimizer.learning_rate), step)

            # Validation step
            if step % args.validation_step == 0:
                if "validation_dataset" in locals():
                    val_losses = []
                    for val_batch_idx in range(validation_dataset.num_batches):
                        (queries, documents), labels = validation_dataset[val_batch_idx]
                        val_loss = trainer.test_on_batch(
                            x=[queries, documents],
                            y=labels,
                        )
                        val_losses.append(val_loss)
                    mean_val_loss = np.mean(val_losses)
                    with train_summary_writer.as_default():
                        tf.summary.scalar("Valiation_Loss", mean_val_loss, step)
                    if mean_val_loss < best_loss:
                        best_loss = mean_val_loss
                        tf.saved_model.save(hub_load, os.path.join(args.save_dir, args.save_name, "best_model"))
                        with open(os.path.join(args.save_dir, args.save_name, "best_losses.txt"), "a") as f:
                            f.write(f"{best_loss}\n")
                else:
                    tf.saved_model.save(hub_load, os.path.join(args.save_dir, args.save_name, "best_model"))
        training_dataset.on_epoch_end()
        K.set_value(trainer.optimizer.learning_rate, K.get_value(trainer.optimizer.learning_rate) * (1 - args.learning_rate_decay))

    # Validation step
    if "validation_dataset" in locals():
        val_losses = []
        for val_batch_idx in range(validation_dataset.num_batches):
            (queries, documents), labels = validation_dataset[val_batch_idx]
            val_loss = trainer.test_on_batch(
                x=[queries, documents],
                y=labels,
            )
            val_losses.append(val_loss)
        mean_val_loss = np.mean(val_losses)
        with train_summary_writer.as_default():
            tf.summary.scalar("Valiation_Loss", mean_val_loss, step)
        if mean_val_loss < best_loss:
            best_loss = mean_val_loss
            tf.saved_model.save(hub_load, os.path.join(args.save_dir, args.save_name, "best_model"))
            with open(os.path.join(args.save_dir, args.save_name, "best_losses.txt"), "a") as f:
                f.write(f"{best_loss}\n")
    else:
        tf.saved_model.save(hub_load, os.path.join(args.save_dir, args.save_name, "best_model"))