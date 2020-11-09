import json
import sys

data = None
with open('data/en_wn_full/all.json') as fin:
    data = json.load(fin)

if not data:
    print('oof')

to_out = {}
count = 0
for word, defin in data.items():
    # include the proper noun filter as well
    if word[0].isupper():
        continue

        # if 'she' is not within 5 words of 'he', replace with 'he or she'
        # if 'her' is not within 5 words of 'his', replace with 'his or her'
    new_defin = []

    for i, x in enumerate(defin[0]):
        if x == 'he':
            if 'she' not in defin[0][i-5:i+5]:
                new_defin += ['he', 'or', 'she']
                count += 1
                continue
        if x == 'his':
            if 'her' not in defin[0][i-5:i+5]:
                new_defin += ['his', 'or', 'her']
                count += 1
                continue
        new_defin += [x]
    to_out[word] = [new_defin]

print(json.dumps(to_out, indent=2))
