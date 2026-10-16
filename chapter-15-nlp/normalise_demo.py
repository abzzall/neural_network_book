"""Code points, UTF-8 bytes, Unicode normalisation, and case folding."""

import unicodedata


def describe(text):
    points = ' '.join(f'U+{ord(char):04X}' for char in text)
    return f'{len(text)} code points, {len(text.encode("utf-8"))} bytes: {points}'


composed = '\u00e9'        # e-acute as a single code point
decomposed = 'e\u0301'     # e followed by a combining acute accent
print('composed:  ', describe(composed))
print('decomposed:', describe(decomposed))
print('equal as strings:', composed == decomposed)
print('equal after NFC:',
      unicodedata.normalize('NFC', composed) == unicodedata.normalize('NFC', decomposed))

ligature = '\ufb01le \uff12\uff10\uff12\uff16'   # "fi" ligature, full-width digits
print('NFC keeps compatibility forms:', unicodedata.normalize('NFC', ligature) == ligature)
print('NFKC result:', unicodedata.normalize('NFKC', ligature))

kazakh = '\u049a\u0430\u0437\u0430\u049b\u0441\u0442\u0430\u043d'   # "Kazakhstan" in Kazakh
print('Kazakh word:', describe(kazakh))
print('first letter after casefold:', describe(kazakh.casefold()[0]))
