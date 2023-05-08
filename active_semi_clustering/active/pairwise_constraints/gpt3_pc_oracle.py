import json
import jsonlines
import os
import time

import openai

from .example_oracle import MaximumQueriesExceeded

class GPT3Oracle:
    def __init__(self, X, labels, max_queries_cnt=2000, side_information=None, read_only=False):
        self.labels = labels
        self.queries_cnt = 0
        self.max_queries_cnt = max_queries_cnt

        self.side_information = side_information
        self.cache_dir = "/projects/ogma1/vijayv/okb-canonicalization/clustering/file/gpt3_cache"
        self.cache_file = os.path.join(self.cache_dir, "pairwise_constraint_cache.jsonl")
        if os.path.exists(self.cache_file):
            self.cache_rows = list(jsonlines.open(self.cache_file))
        else:
            self.cache_rows = []
        if not read_only:
            self.cache_writer = jsonlines.open(self.cache_file, mode='a')
        else:
            self.cache_writer = jsonlines.open(self.cache_file, mode='r')
        self.NUM_RETRIES = 2
        self.read_only = read_only

        side_info = self.side_information.side_info
        self.sentence_unprocessing_mapping_file = os.path.join(self.cache_dir, "sentence_unprocessing_map.json")
        sentence_unprocessing_mapping = json.load(open(self.sentence_unprocessing_mapping_file))
        selected_sentences = []
        ents = []

        for i in range(len(X)):
            ents.append(side_info.id2ent[i])
            entity_sentence_idxs = side_info.ent_id2sentence_list[i]
            unprocessed_sentences = [sentence_unprocessing_mapping[side_info.sentence_List[j]] for j in entity_sentence_idxs]
            entity_sentences = self.process_sentence_punctuation(unprocessed_sentences)

            '''
            Choose longest sentence under 306 characers, as in
            https://github.com/Yang233666/cmvc/blob/6e752b1aa5db7ff99eb2fa73476e392a00b0b89a/Context_view.py#L98
            '''
            longest_sentences = sorted([s for s in entity_sentences if len(s) < 599], key=len, reverse=True)
            if len(longest_sentences) == 0:
                breakpoint()
            selected_sentences.append([longest_sentences[0]])

        self.ents = ents
        self.selected_sentences = selected_sentences

        self.gpt3_pairwise_labels = {}
        for row in self.cache_rows:
            self.gpt3_pairwise_labels[(row["entity1"], row["entity2"])] = row["label"]

    def process_sentence_punctuation(self, sentences):
        processed_sentence_set = []
        for s in sentences:
            processed_sentence_set.append(s.replace("-LRB-", "(").replace("-RRB-", ")"))
        return processed_sentence_set

    def construct_pairwise_oracle_prompt(self, i, j):
        breakpoint()

    def query(self, i, j):
        if self.queries_cnt < self.max_queries_cnt:
            self.queries_cnt += 1

            prompt = self.construct_pairwise_oracle_prompt(i, j)

            pair_label = None

            failure = True
            num_retries = 0
            while failure and num_retries < self.NUM_RETRIES:
                cache_row = None
                try:
                    start = time.perf_counter()
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "user", "content": prompt},
                        ],
                    )
                    message = json.loads(str(response.choices[0]))["message"]["content"]
                    try:
                        pair_label = int(message.strip() == "False")
                        cache_row = {"entity1": self.ents[i],
                                     "entity1": self.ents[j],
                                     "label": pair_label}
                        self.cache_writer.write(cache_row)
                        failure = False
                    except:
                        time.sleep(0.8)
                    num_retries += 1
                    end = time.perf_counter()
                    if end - start < 1:
                        time.sleep(1 - (end - start))
                except:
                    time.sleep(3)

            if pair_label is None:
                return False
            else:
                pair_label
        else:
            raise MaximumQueriesExceeded