"""How an English subword vocabulary handles an English and a Kazakh sentence."""

from transformers import AutoTokenizer

from encoder import CACHE, load_encoder

load_encoder()                                    # explains how to download if missing
tokenizer = AutoTokenizer.from_pretrained(CACHE, local_files_only=True)
sentences = {
    'English': 'Kazakhstan is a country in Central Asia.',
    # The same statement in Kazakh, written with escapes to keep this file ASCII.
    'Kazakh': ('\u049a\u0430\u0437\u0430\u049b\u0441\u0442\u0430\u043d '
               '\u041e\u0440\u0442\u0430\u043b\u044b\u049b '
               '\u0410\u0437\u0438\u044f\u0434\u0430\u0493\u044b '
               '\u043c\u0435\u043c\u043b\u0435\u043a\u0435\u0442.'),
}
for language, sentence in sentences.items():
    tokens = tokenizer.tokenize(sentence)
    unknown = tokens.count(tokenizer.unk_token)
    print(f'{language}: {len(sentence.split())} words -> {len(tokens)} tokens, '
          f'{unknown} unknown')
    print('  ' + ' '.join(tokens))
