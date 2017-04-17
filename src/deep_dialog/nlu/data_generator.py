import json
import random
from copy import copy
import numpy as np


def template_to_BIO(template):
    markup = []
    for token in template:
        if token[0] == token[-1] == "$":
            markup.append("B-" + token.split("_")[0][1:])
        else:
            markup.append("O")
    return markup


class NLUDataGenerator:
    def __init__(self, path_to_template, path_to_dict, path_to_slot, seq_len=64, batch_size=32):
        self.batch_size = batch_size
        self.seq_len = seq_len
        with open(path_to_dict, "rt") as f_dict:
            self.dict = json.load(f_dict)
        with open(path_to_template, "rt") as f_tmpl:
            templates = json.load(f_tmpl)
        self.slots = []
        with open(path_to_slot, "rt") as f_slot:
            for line in f_slot:
                self.slots.append(line.strip())
        self.tag_count = len(self.slots) + 2

        self.templates = []
        for t in templates:
            agent_line = t["nl"]["agt"].split()
            self.templates.append((agent_line, template_to_BIO(agent_line), "agt"))

            user_line = t["nl"]["usr"].split()
            self.templates.append((user_line, template_to_BIO(user_line), "usr"))

    def __next__(self):
        batch = random.sample(self.templates, self.batch_size)
        filled_batch = []
        for t, m, role in batch:
            input_ = []
            target = []
            for i in range(len(m)):
                target.append(m[i])
                if m[i] == 'O':
                    input_.append(t[i])
                else:  # m[i] is B-smth
                    tag = m[i].split("-")[1]
                    filler = random.choice(self.dict[tag]).split()
                    input_.append(filler[0])
                    for f in filler[1:]:
                        target.append("I-" + tag)
                        input_.append(f)
            filled_batch.append((copy(input_), copy(target)))
        return filled_batch

    def digitize_batch(self, batch):
        dbatch = []
        for row in batch:
            drow = np.zeros((self.seq_len,))
            for i, token in enumerate(row):
                if token == "O":
                    drow[i] = 1
                else:
                    tag, slot = token.split("_")
                    drow[i] = 2 + self.slots.index(slot) * 2 + 0 if tag == "B" else 1
            dbatch.append(drow)
        return np.stack(dbatch)
