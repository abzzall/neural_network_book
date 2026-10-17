"""The Python the book uses: values, lists, dictionaries, loops, functions, classes."""

prices = [3.5, 2.0, 4.25]                       # a list: ordered and changeable
prices.append(1.25)
print('list:', prices, 'length', len(prices), 'last', prices[-1])

stock = {'apples': 12, 'pears': 0}             # a dictionary: key -> value
stock['plums'] = 7
available = [name for name, count in stock.items() if count > 0]   # comprehension
print('available:', available)

total = 0.0
for i, price in enumerate(prices):              # index and value together
    total += price
print(f'total {total:.2f} over {i + 1} items')


def mean(values, default=0.0):
    """A function with a default argument and a docstring."""
    return sum(values) / len(values) if values else default


print('mean:', mean(prices), 'mean of nothing:', mean([]))


class Counter:
    """A class bundles data (attributes) with behaviour (methods)."""

    def __init__(self):
        self.counts = {}

    def add(self, word):
        self.counts[word] = self.counts.get(word, 0) + 1


c = Counter()
for word in 'the cat sat on the mat'.split():
    c.add(word)
print('counts:', c.counts)
