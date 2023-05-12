import json
import jsonlines
import math
import os
import time

import openai

from .example_oracle import MaximumQueriesExceeded

class GPT3Oracle:
    def __init__(self, X, labels, max_queries_cnt=2000, num_predictions=50, side_information=None, read_only=False):
        self.labels = labels
        self.queries_cnt = 0
        self.max_queries_cnt = max_queries_cnt
        self.num_predictions = num_predictions

        self.side_information = side_information
        self.cache_dir = "/projects/ogma1/vijayv/okb-canonicalization/clustering/file/gpt3_cache"
        self.cache_file = os.path.join(self.cache_dir, "pairwise_constraint_cache_multi_predictions.jsonl")
        if os.path.exists(self.cache_file):
            self.cache_rows = list(jsonlines.open(self.cache_file))
        else:
            self.cache_rows = []
        if not read_only:
            self.cache_writer = jsonlines.open(self.cache_file, mode='a', flush=True)
        else:
            self.cache_writer = jsonlines.open(self.cache_file, mode='r', flush=True)
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
            sorted_pair_list = sorted([row["entity1"], row["entity2"]])
            self.gpt3_pairwise_labels[tuple(sorted_pair_list)] = row["labels"]

    def process_sentence_punctuation(self, sentences):
        processed_sentence_set = []
        for s in sentences:
            processed_sentence_set.append(s.replace("-LRB-", "(").replace("-RRB-", ")"))
        return processed_sentence_set


    def construct_single_example(self, i, j, add_label=True):
        template_prefix = f"""1) {self.ents[i]}
Context Sentence: "{self.selected_sentences[i][0]}"
2) {self.ents[j]}
Context Sentence: "{self.selected_sentences[j][0]}"
Given this context, would {self.ents[i]} and {self.ents[j]} link to the same entity's article on Wikipedia? """
        if add_label:
            if self.labels[i] == self.labels[j]:
                label = "Yes"
            else:
                label = "No"
            full_example = template_prefix + label
            return full_example
        else:
            return template_prefix

    def construct_pairwise_oracle_prompt(self, i, j):
        side_info = self.side_information.side_info
        example_1 = self.construct_single_example(side_info.ent2id["B.A"], side_info.ent2id["M.D."])
        example_2 = self.construct_single_example(side_info.ent2id["B.A"], side_info.ent2id["bachelor"])
        example_3 = self.construct_single_example(side_info.ent2id["British Government"], side_info.ent2id["government"])
        example_4 = self.construct_single_example(side_info.ent2id["Duke of York"], side_info.ent2id["Frederick"])
        prefix = "\n\n".join([example_1, example_2, example_3, example_4])
        filled_template = self.construct_single_example(i, j, add_label = False)
        return prefix + "\n\n" + filled_template

    @staticmethod
    def filter_high_entropy_predictions(pair_labels, majority_class_threshold=0.999999):
        '''If the majority class probability is < `majority_class_threshold`, return None'''
        assert None not in pair_labels
        p = sum(pair_labels) / len(pair_labels)

        if p > 0.5 and p > majority_class_threshold:
            return True
        elif p < 0.5 and p < 1 - majority_class_threshold:
            return False
        else:
            return None

    def query(self, i, j):
        if self.queries_cnt < self.max_queries_cnt:
            self.queries_cnt += 1
            sorted_pair_list = sorted([self.ents[i], self.ents[j]])
            sorted_pair = tuple(sorted_pair_list)
            if  sorted_pair in self.gpt3_pairwise_labels:
                return self.filter_high_entropy_predictions(self.gpt3_pairwise_labels[sorted_pair])

            prompt = self.construct_pairwise_oracle_prompt(i, j)

            pair_labels_not_none = []

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
                        temperature=1.0,
                        max_tokens=1,
                        n=50,
                    )

                    pair_labels = []
                    for choice in response.choices:
                        message = json.loads(str(choice))["message"]["content"]
                        if message.strip() == "Yes":
                            pair_label = True
                        elif message.strip() == "No":
                            pair_label = False
                        else:
                            pair_label = None
                        pair_labels.append(pair_label)


                    pair_labels_not_none = [x for x in pair_labels if x is not None]
                    if len(pair_labels_not_none) <= self.num_predictions / 2:
                        time.sleep(0.8)
                    else:
                        cache_row = {"entity1": self.ents[i],
                                     "entity2": self.ents[j],
                                     "labels": pair_labels_not_none,
                                     "p_true": round(sum(pair_labels_not_none) / len(pair_labels_not_none), 4)}
                        self.cache_writer.write(cache_row)
                        self.gpt3_pairwise_labels[sorted_pair] = pair_labels_not_none
                        failure = False


                    num_retries += 1
                    end = time.perf_counter()
                    if end - start < 1:
                        time.sleep(1 - (end - start))
                except:
                    time.sleep(3)

            return self.filter_high_entropy_predictions(pair_labels_not_none)
        else:
            raise MaximumQueriesExceeded