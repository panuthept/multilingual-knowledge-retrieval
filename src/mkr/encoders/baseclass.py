from abc import abstractmethod


@abstractmethod
class SentenceEncoderBase:
    def encode(self, *args, **kwargs):
        raise NotImplementedError

    def encode_batch(self, *args, **kwargs):
        raise NotImplementedError