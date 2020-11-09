import json

data = None
with open('data/en_wn_full/all.json') as fin:
    data = json.load(fin)

if not data:
    print('oof')

to_out = {}
for word, defin in data.items():
    if word[0].isupper():
        continue

    to_out[word] = defin


print(json.dumps(to_out, indent=2))
