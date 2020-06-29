import unittest
from src.transformer import *


class TestMultiHeadAttention(unittest.TestCase):

    def test_attention(self):
        batch_size = 100
        sent_len = 30
        nheads = 8
        d_key = 64

        mhatt = MultiHeadAttention(nheads*d_key, nheads, masked=True)

        A = torch.ones((batch_size, sent_len, nheads, d_key))
        B = torch.ones((batch_size, int(sent_len/2), nheads, d_key))

        ret, att = mhatt.attention(A, B, B)

        self.assertEqual((batch_size, nheads, sent_len, int(sent_len/2)), att.shape)
        self.assertEqual((batch_size, sent_len, nheads, d_key), ret.shape)

    def test_mhattention(self):
        batch_size = 100
        sent_len = 30
        nheads = 8
        d_key = 64
        d_model = nheads*d_key

        mhatt = MultiHeadAttention(d_model, nheads, masked=True)

        A = torch.ones((batch_size, sent_len, d_model))
        B = torch.ones((batch_size, int(sent_len / 2), d_model))

        ret = mhatt(A, B, B)

        self.assertEqual((batch_size, sent_len, d_model), ret.shape)


class TestEmbedding(unittest.TestCase):

    def setUp(self):
        batch_size = 100
        sent_len = 30
        vocab_size = 1000
        model_dim = 512

        self.encoder_shape = (batch_size, sent_len, model_dim)
        self.decoder_shape = (batch_size, sent_len, vocab_size)

        self.model = Embedding(vocab_size, model_dim)

        self.input_a = torch.ones((batch_size, sent_len), dtype=torch.long)
        self.input_b = torch.ones((batch_size, sent_len, model_dim))

        self.target_a = torch.ones((batch_size, sent_len, model_dim)) * 10
        self.target_b = torch.ones((batch_size, sent_len, vocab_size)) * 10

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)

        self.weights_in = deepcopy(self.model.encoder.weight)
        self.weights_out = deepcopy(self.model.decoder.weight)

    def test_encoder(self):
        self.optimizer.zero_grad()
        output_a = self.model(self.input_a)
        loss = torch.sum(output_a-self.target_a)

        loss.backward()
        self.optimizer.step()

        self.assertNotEqual(torch.sum(self.weights_in), torch.sum(self.model.encoder.weight))
        self.assertNotEqual(torch.sum(self.weights_out), torch.sum(self.model.decoder.weight))
        self.assertEqual(torch.sum(self.model.encoder.weight), torch.sum(self.model.decoder.weight))
        self.assertEqual(self.encoder_shape, output_a.shape)

    def test_decoder(self):
        self.optimizer.zero_grad()
        output_b = self.model(self.input_b)
        loss = torch.sum(output_b - self.target_b)

        loss.backward()
        self.optimizer.step()

        self.assertNotEqual(torch.sum(self.weights_in), torch.sum(self.model.encoder.weight))
        self.assertNotEqual(torch.sum(self.weights_out), torch.sum(self.model.decoder.weight))
        self.assertEqual(torch.sum(self.model.encoder.weight), torch.sum(self.model.decoder.weight))
        self.assertEqual(self.decoder_shape, output_b.shape)


class TestPositionalEncoder(unittest.TestCase):

    def test_pe(self):
        model_dim = 512
        max_len = 10000
        batch_size = 100
        sent_len = 30

        A = torch.ones((batch_size, sent_len, model_dim))
        pe = PositionalEncoder(model_dim, max_len)

        ret = pe(A)


class TestEncoderLayer(unittest.TestCase):

    def test_encoder(self):
        model_dim = 64
        hidden_dim = 10
        nheads = 4
        batch_size = 100
        sent_len = 30

        A = torch.ones((batch_size, sent_len, model_dim))
        encoder = EncoderLayer(model_dim, hidden_dim, nheads)

        enc = encoder(A)

        self.assertEqual((batch_size, sent_len, model_dim),enc.shape)


class TestDecoderLayer(unittest.TestCase):

    def test_decoder(self):
        model_dim = 64
        hidden_dim = 10
        nheads = 4
        batch_size = 100
        sent_len = 30

        A = torch.ones((batch_size, sent_len, model_dim))
        B = torch.ones((batch_size, sent_len//2, model_dim))
        decoder = DecoderLayer(model_dim, hidden_dim, nheads)

        dec = decoder(A, B)

        self.assertEqual((batch_size, sent_len, model_dim), dec.shape)


class TestTransformer(unittest.TestCase):

    def test_transformer(self):

        model_dim = 512
        batch_size = 300
        sent_len = 30
        vocab_size = 100
        hidden_dim = 200
        nheads = 8
        max_len = sent_len*100
        depth = 1

        t = Transformer(vocab_size, model_dim, hidden_dim, nheads, max_len, depth)

        x = torch.ones((batch_size, sent_len),dtype=torch.long)
        y1 = torch.ones((batch_size, sent_len//2), dtype=torch.long)

        ret = t(x, y1)

        self.assertEqual((batch_size, sent_len//2, vocab_size), ret.shape)
