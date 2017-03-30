import json


class NLUDataGenerator:
    def __init__(self, path_to_template, path_to_dict):
        with open(path_to_dict, "wt") as f_dict:
            self.dict = json.load(f_dict)
        with open(path_to_template, "wt") as f_tmpl:
            templates = json.load(f_tmpl)

        self.templates = []
        for t in templates:
            self.templates.append((t["nl"]["agt"], "agt"))
            self.templates.append((t["nl"]["usr"], "usr"))

    def __next__(self):
        pass

