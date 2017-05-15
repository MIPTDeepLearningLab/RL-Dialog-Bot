import json
import random
import copy
import numpy as np
import pandas as pd

class NLUDataGenerator:
    def __init__(self, path_to_template, path_to_dict, path_to_slot, path_to_acts, seq_len=64, batch_size=32):
        
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.dict = read_json(path_to_dict) # for random choices and slot filling;
        self.slots = list(np.array(pd.read_csv(path_to_slot, sep = '\n')).reshape(-1))
        self.acts = list(np.array(pd.read_csv(path_to_acts, sep = '\n')).reshape(-1))
        self.spec_no_slots = ['bye', 'hello', 'reqalts', 'doncare'] # all template is marked always;
        self.spec_with_slots = ['affirm', 'negate'] # if there's no slots all template is marked;
        templates = pd.read_csv(path_to_template).fillna(-1)

        self.templates = [] # for templates with blanks;
        self.vocab = set() # all the words in all templates;
        for index, row in templates.iterrows():
            
            user_line = row["nl"]
            acts = [row["act"+str(i)] for i in range(1,4) if row["act"+str(i)]!=-1]
            bio_slots, bio_acts, fill_slots = self.template_to_BIO(user_line, acts)
            self.templates.append((user_line, bio_slots, bio_acts, fill_slots))

            for w in user_line.split():
                if w[0] == w[-1] == "$":
                    continue
                self.vocab.add(w)

        for type_ in ['informable', 'requestable']:
            for tag in self.dict[type_]:
                for filling in self.dict[type_][tag]:
                    for w in filling.split(): # Конечно, это вызывает вопросы, т.к имена будут явно разнесены(
                        self.vocab.add(w)
        
        self.vocab = dict(zip(self.vocab, range(1, len(self.vocab) + 1)))
        
    def template_to_BIO(self, template, acts):

        bio_slots = []
        bio_acts = []
        acts_iter = cycle(acts)
        fill_slots = True

        # special case:
        if len(acts) == 1:
            if (acts[0] in self.spec_no_slots) or \
            (('$' not in template) and (acts[0] in self.spec_with_slots)):
                nl_len = len(template.split())
                bio_slots.extend(['B-'+acts[0]]*nl_len) # slots have the same value as act;
                bio_acts.extend(['B-'+acts[0]]*nl_len)
                fill_slots = False
                return bio_slots, bio_acts, fill_slots
        # else:
        for token in template.split():
            if token[0] == token[-1] == "$":
                slot = token[1:-1]
                bio_slots.append('B-'+slot)
                bio_acts.append('B-'+next(acts_iter))
            else:
                bio_slots.append("O")
                bio_acts.append("O")
        return bio_slots, bio_acts, fill_slots

    def __next__(self):
        batch = random.sample(self.templates, self.batch_size)
        filled_batch = []
        for nl, slot_arr, act_arr, fill_slots in batch:
            nl_arr = nl.split()
            input_ = []
            target_slot = []
            target_acts = []
            
            if not fill_slots:
                input_.extend(nl_arr)
                target_slot.extend(slot_arr)
                target_acts.extend(act_arr)
                continue
                
            for i in range(len(nl_arr)):
                if slot_arr[i] == 'O':
                    input_.append(nl_arr[i])
                    target_slot.append(slot_arr[i])
                    target_acts.append(act_arr[i])
                else:  # slot_arr[i] is B-smth
                    slot = slot_arr[i].split("-")[1]
                    act = act_arr[i].split("-")[1]
                    if act == 'request':
                        filler = random.choice(self.dict['requestable'][slot]).split()
                    else:
                        filler = random.choice(self.dict['informable'][slot]).split()
                    input_.append(filler[0])
                    target_slot.append(slot_arr[i])
                    target_acts.append(act_arr[i])
                    for f in filler[1:]:
                        target_slot.append("I-" + slot)
                        target_acts.append("I-" + act)
                        input_.append(f)
            filled_batch.append((copy.deepcopy(input_), copy.deepcopy(target_slot), copy.deepcopy(target_acts)))
        return filled_batch

    # TODO: need right digitilazer;
    # this is draft version:
    # def digitize_batch(self, batch):  # TODO: digitize words too
    #     dbatch = []
    #     for row in batch:
    #         drow = np.zeros((self.seq_len,))
    #         for i, token in enumerate(row):
    #             if token == "O":
    #                 drow[i] = 1
    #             else:
    #                 tag, slot = token.split("-")
    #                 drow[i] = 2 + self.slots.index(slot) * 2 + 0 if tag == "B" else 1
    #         dbatch.append(drow)
    #     return np.stack(dbatch)