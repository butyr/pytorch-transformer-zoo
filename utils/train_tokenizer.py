from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import Lowercase, NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

path_data = "../../ml-datasets/wmt14/tokenizer/"

path_train_src = "../../ml-datasets/wmt14/train.en"
path_train_tgt = "../../ml-datasets/wmt14/train.de"

tokenizer = Tokenizer(BPE())
tokenizer.normalizer = Sequence([
    NFKC(),
    Lowercase()
])

tokenizer.pre_tokenizer = ByteLevel()
tokenizer.decoder = ByteLevelDecoder()

trainer = BpeTrainer(vocab_size=25000, show_progress=True, initial_alphabet=ByteLevel.alphabet(),
                     min_frequency=2, special_tokens=["<pad>", "<s>", "</s>", "<unk>", "<mask>", ])
tokenizer.train(trainer, [path_train_src, path_train_tgt])

print("Trained vocab size: {}".format(tokenizer.get_vocab_size()))

tokenizer.model.save(path_data)
