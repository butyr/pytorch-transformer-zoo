import unittest
from src.transformer import *


d_key = 4
nheads = 2
model_dim = d_key*nheads
batch_size = 4
sent_len = 4
hidden_dim = 16
vocab_size = 14

max_len = sent_len * 100
depth = 2

torch.manual_seed(1234)


class TestTensorShapes(unittest.TestCase):

    def test_attention(self):
        mhatt = MultiHeadAttention(nheads*d_key, nheads, masked=True)

        A = torch.ones((batch_size, sent_len, nheads, d_key))
        B = torch.ones((batch_size, int(sent_len/2), nheads, d_key))

        ret, att = mhatt.attention(A, B, B)

        self.assertEqual(
            (batch_size, nheads, sent_len, int(sent_len/2)), att.shape
        )
        self.assertEqual(
            (batch_size, sent_len, nheads, d_key), ret.shape
        )

    def test_multi_head_attention(self):
        mhatt = MultiHeadAttention(model_dim, nheads, masked=True)

        A = torch.ones((batch_size, sent_len, model_dim))
        B = torch.ones((batch_size, sent_len // 2, model_dim))

        ret = mhatt(A, B, B)

        self.assertEqual((batch_size, sent_len, model_dim), ret.shape)

    def test_transformer(self):
        model = Transformer(
            vocab_size,
            model_dim,
            hidden_dim,
            nheads,
            max_len,
            depth
        )

        x = torch.ones(
            (batch_size, sent_len), dtype=torch.long
        )
        y = torch.ones(
            (batch_size, sent_len//2), dtype=torch.long
        )

        ret = model(x, y)

        self.assertEqual(
            (batch_size, sent_len//2, vocab_size), ret.shape
        )

    def test_pe(self):
        A = torch.ones((batch_size, sent_len, model_dim))
        pe = PositionalEncoder(model_dim, max_len)

        ret = pe(A)

        self.assertEqual((batch_size, sent_len, model_dim), ret.shape)

    def test_encoder_layer(self):
        A = torch.ones((batch_size, sent_len, model_dim))
        encoder = EncoderLayer(model_dim, hidden_dim, nheads)

        enc = encoder(A)

        self.assertEqual((batch_size, sent_len, model_dim), enc.shape)

    def test_decoder_layer(self):
        A = torch.ones((batch_size, sent_len, model_dim))
        B = torch.ones((batch_size, sent_len//2, model_dim))
        decoder = DecoderLayer(model_dim, hidden_dim, nheads)

        dec, _ = decoder(A, B)

        self.assertEqual((batch_size, sent_len, model_dim), dec.shape)


class TestEmbedding(unittest.TestCase):

    def setUp(self):
        self.encoder_shape = (batch_size, sent_len, model_dim)
        self.decoder_shape = (batch_size, sent_len, vocab_size)

        self.model = Embedding(vocab_size, model_dim)

        self.input_a = torch.ones((batch_size, sent_len), dtype=torch.long)
        self.input_b = torch.ones((batch_size, sent_len, model_dim))

        self.target_a = torch.ones(
            (batch_size, sent_len, model_dim)
        )
        self.target_b = torch.ones(
            (batch_size, sent_len, vocab_size)
        )

        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=0.01, momentum=0.9
        )

        self.weights_in = deepcopy(self.model.encoder.weight)
        self.weights_out = deepcopy(self.model.decoder.weight)

    def test_encoder_tie_weights(self):
        self.optimizer.zero_grad()
        output_a = self.model(self.input_a)
        loss = torch.sum(output_a-self.target_a)

        loss.backward()
        self.optimizer.step()

        self.assertNotEqual(
            torch.sum(self.weights_in),
            torch.sum(self.model.encoder.weight)
        )
        self.assertNotEqual(
            torch.sum(self.weights_out),
            torch.sum(self.model.decoder.weight)
        )
        self.assertEqual(
            torch.sum(self.model.encoder.weight),
            torch.sum(self.model.decoder.weight)
        )
        self.assertEqual(self.encoder_shape, output_a.shape)

    def test_decoder_tie_weights(self):
        self.optimizer.zero_grad()
        output_b = self.model(self.input_b, inverse=True)
        loss = torch.sum(output_b - self.target_b)

        loss.backward()
        self.optimizer.step()

        self.assertNotEqual(
            torch.sum(self.weights_in),
            torch.sum(self.model.encoder.weight)
        )
        self.assertNotEqual(
            torch.sum(self.weights_out),
            torch.sum(self.model.decoder.weight)
        )
        self.assertEqual(
            torch.sum(self.model.encoder.weight),
            torch.sum(self.model.decoder.weight)
        )
        self.assertEqual(self.decoder_shape, output_b.shape)


class TestSanityChecks(unittest.TestCase):

    def test_init_loss(self):
        """Compares loss@init with theoretical loss."""

        model = Transformer(
            vocab_size,
            model_dim,
            hidden_dim,
            nheads,
            max_len,
            depth
        )

        inputs = torch.ones(
            (batch_size, sent_len), dtype=torch.long
        )

        targets = torch.ones(
            (batch_size, sent_len), dtype=torch.long
        )

        outputs = model(inputs, targets)

        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(
            outputs.reshape(-1, vocab_size),
            targets.reshape(-1)
        )

        self.assertAlmostEqual(
            -np.log(1./vocab_size), loss.detach().numpy(), delta=0.2
        )

    def test_softmax_dim(self):

        model = Transformer(
            vocab_size,
            model_dim,
            hidden_dim,
            nheads,
            max_len,
            depth
        )

        inputs = torch.ones(
            (batch_size, sent_len), dtype=torch.long
        )

        targets = torch.ones(
            (batch_size, sent_len), dtype=torch.long
        )

        outputs = model(inputs, targets)

        for i in range(batch_size):
            for j in range(sent_len):
                self.assertAlmostEqual(
                    1.0,
                    outputs[i, j, :].sum().detach().numpy(),
                    places=6
                )


class TestGradientFlows(unittest.TestCase):

    def test_one_step_grad(self):
        """Tests whether all parameters are updated."""

        model = Transformer(
            vocab_size,
            model_dim,
            hidden_dim,
            nheads,
            max_len,
            depth
        )

        inputs = torch.ones(
            (batch_size, sent_len), dtype=torch.long
        )
        targets = torch.zeros(
            (batch_size, sent_len // 2), dtype=torch.long
        )

        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.01, momentum=0.9
        )
        model_t0 = deepcopy(model)

        optimizer.zero_grad()
        outputs = model(inputs, targets)
        targets = torch.argmax(
            (outputs*(-1)).reshape(-1, vocab_size), dim=-1
        )

        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(
            outputs.reshape(-1, vocab_size),
            targets
        )

        loss.backward()
        optimizer.step()

        for p0, p in zip(
                model_t0.named_parameters(),
                model.named_parameters()
        ):
            self.assertNotEqual(
                0.0,
                torch.sum(torch.square(p[1]-p0[1]))
            )


    def test_batch_dim(self):
        """Tests consistency of batch dimension."""

        model = Transformer(
            vocab_size,
            model_dim,
            hidden_dim,
            nheads,
            max_len,
            depth
        )

        inputs = torch.ones(
            (batch_size, sent_len), dtype=torch.long
        )

        targets = torch.ones(
            (batch_size, sent_len), dtype=torch.long
        )

        for i in range(batch_size):
            outputs = model(inputs, targets)

            loss = outputs[i, :, :].sum()
            grad = torch.autograd.grad(loss, model.src_embedding)[0]

            self.assertNotEqual(0.0, loss)

            self.assertEqual(
                0.0,
                grad[:i, :, :].sum()
            )
            self.assertEqual(
                0.0,
                grad[i+1:, :, :].sum()
            )
            self.assertNotEqual(
                0.0,
                grad[i, :, :].sum()
            )

    def test_mask(self):
        """Tests masking of decoder inputs."""

        model = Transformer(
            vocab_size,
            model_dim,
            hidden_dim,
            nheads,
            max_len,
            depth
        )

        inputs = torch.ones(
            (batch_size, sent_len), dtype=torch.long
        )

        targets = torch.ones(
            (batch_size, sent_len), dtype=torch.long
        )

        for i in range(sent_len):
            outputs = model(inputs, targets)

            loss = outputs[:, i, :].sum()
            grad = torch.autograd.grad(loss, model.tgt_embedding)[0]

            self.assertNotEqual(0.0, loss)
            self.assertNotEqual(0.0, grad.sum())

            self.assertEqual(
                0.0,
                grad[:, i+1:, :].sum()
            )
            self.assertNotEqual(
                0.0,
                grad[:, i, :].sum()
            )
            self.assertNotEqual(
                0.0,
                grad[:, :i+1, :].sum()
            )
