import json
import jsonlines
import math
import os
import time

import openai

from .example_oracle import MaximumQueriesExceeded

class GPT3Oracle:
    def __init__(self, X, labels, dataset_name, split=None, max_queries_cnt=2500, num_predictions=5, side_information=None, read_only=False):
        self.labels = labels
        self.queries_cnt = 0
        self.max_queries_cnt = max_queries_cnt
        self.num_predictions = num_predictions

        self.side_information = side_information
        self.cache_dir = "/projects/ogma1/vijayv/okb-canonicalization/clustering/file/gpt3_cache"
        self.dataset_name = dataset_name
        self.cache_file = os.path.join(self.cache_dir, f"{dataset_name}_pairwise_constraint_cache.jsonl")
        if os.path.exists(self.cache_file):
            self.cache_rows = list(jsonlines.open(self.cache_file))
        else:
            self.cache_rows = []
        if not read_only:
            self.cache_writer = jsonlines.open(self.cache_file, mode='a', flush=True)
        else:
            self.cache_writer = jsonlines.open(self.cache_file, mode='r')

        self.NUM_RETRIES = 2
        self.read_only = read_only

        side_info = self.side_information.side_info
        self.sentence_unprocessing_mapping_file = os.path.join(self.cache_dir, f"{dataset_name}_{split}_sentence_unprocessing_map.json")
        sentence_unprocessing_mapping = json.load(open(self.sentence_unprocessing_mapping_file))
        selected_sentences = []
        ents = []

        for i in range(len(X)):
            ents.append(side_info.id2ent[i])
            entity_sentence_idxs = side_info.ent_id2sentence_list[i]
            unprocessed_sentences = [sentence_unprocessing_mapping[side_info.sentence_List[j]] for j in entity_sentence_idxs]
            entity_sentences = self.process_sentence_punctuation(unprocessed_sentences)
            entity_sentences_dedup = list(set(entity_sentences))

            '''
            Choose longest sentence under 306 characers, as in
            https://github.com/Yang233666/cmvc/blob/6e752b1aa5db7ff99eb2fa73476e392a00b0b89a/Context_view.py#L98
            '''
            longest_sentences = sorted([s for s in entity_sentences_dedup if len(s) < 599], key=len)
            selected_sentences.append(list(set(longest_sentences[:3])))

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
        context_labels = ["a", "b", "c", "d"]
        context_1 = "\n".join([context_labels[ind] + ") " + '"' + sent + '"' for ind, sent in enumerate(self.selected_sentences[i])])
        context_2 = "\n".join([context_labels[ind] + ") " + '"' + sent + '"' for ind, sent in enumerate(self.selected_sentences[j])])
        if self.dataset_name == "OPIEC59k":
            prompt_suffix = "link to the same entity's article on Wikipedia? "
        elif self.dataset_name == "reverb45k":
            prompt_suffix = "link to the same entity on a knowledge graph like Freebase? "
        else:
            raise NotImplementedError
        template_prefix = f"""1) {self.ents[i]}

Context Sentences:\n{context_1}

2) {self.ents[j]}

Context Sentence:\n{context_2}

Given this context, would {self.ents[i]} and {self.ents[j]} likely {prompt_suffix}"""
        if add_label:
            if self.labels[i] == self.labels[j]:
                label = "Yes"
            else:
                label = "No"
            full_example = template_prefix + label
            return full_example
        else:
            return template_prefix, context_1, context_2

    def construct_pairwise_oracle_prompt(self, i, j):
        side_info = self.side_information.side_info
        if self.dataset_name == "OPIEC59k":
            instruction = """You are tasked with clustering entity strings based on whether they refer to the same Wikipedia article. To do this, you will be given pairs of entity names and asked if their anchor text, if used separately to link to a Wikipedia article, is likely referring to the same article. Entity names may be truncated, abbreviated, or ambiguous.

To help you make this determination, you will be given up to three context sentences from Wikipedia where the entity is used as anchor text for a hyperlink. Amongst each set of examples for a given entity, the entity for all three sentences is a link to the same article on Wikipedia. Based on these examples, you will decide whether the first entity and the second entity listed would likely link to the same Wikipedia article if used as separate anchor text.

Please note that the context sentences may not be representative of the entity's typical usage, but should aid in resolving the ambiguity of entities that have similar or overlapping meanings.

To avoid subjective decisions, the decision should be based on a strict set of criteria, such as whether the entities will generally be used in the same contexts, whether the context sentences mention the same topic, and whether the entities have the same domain and scope of meaning.

Your task will be considered successful if the entities are clustered into groups that consistently refer to the same Wikipedia articles."""
            example_1 = self.construct_single_example(side_info.ent2id["B.A"], side_info.ent2id["M.D."])
            example_2 = self.construct_single_example(side_info.ent2id["B.A"], side_info.ent2id["bachelor"])
            example_3 = self.construct_single_example(side_info.ent2id["Duke of York"], side_info.ent2id["Frederick"])
            example_4 = self.construct_single_example(side_info.ent2id["Academy Award"], side_info.ent2id["Best Actor in Supporting Role"])
            prefix = "\n\n".join([example_1, example_2, example_3, example_4])
        elif self.dataset_name == "reverb45k":
            instruction = """You are tasked with clustering entity strings based on whether they link to the same entity on the Freebase knowledge graph. To do this, you will be given pairs of entity names and asked if these strings, if linked to a knowledge graph, are likely referring to the same entity (e.g. a concept, person, or organization). Entity names may be truncated, abbreviated, or ambiguous.

To help you make this determination, you will be given up to three context sentences from the internet that mention an entity. Amongst each set of examples for a given entity, assume that the entity mentioned in all three context sentences links refers to the same object. Based on these examples, you will decide whether the first entity and the second entity listed are likely to link to the *same* knowledge graph entity.

Please note that the context sentences may not be representative of the entity's typical usage, but should aid in resolving the ambiguity of entities that have similar or overlapping meanings.

To avoid subjective decisions, the decision should be based on a strict set of criteria, such as whether the entities will generally be used in the same contexts, whether the entities likely refer to the same person or organization, whether the context sentences mention the same topic, and whether the entities have the same domain and scope of meaning.

Your task will be considered successful if the entities are clustered into groups that consistently link to the same knowledge graph node."""
            example_1 = self.construct_single_example(side_info.ent2id["Hannibal"], side_info.ent2id["Hannibal Barca"])
            example_2 = self.construct_single_example(side_info.ent2id["Lutheran Church"], side_info.ent2id["Church"])
            example_3 = self.construct_single_example(side_info.ent2id["Grove Art Online"], side_info.ent2id["Oxford Art Online"])
            example_4 = self.construct_single_example(side_info.ent2id["Charlie Williams"], side_info.ent2id["Williams"])
            prefix = "\n\n".join([example_1, example_2, example_3, example_4])
        else:
            raise NotImplementedError
        filled_template, context_1, context_2 = self.construct_single_example(i, j, add_label = False)
        return "\n\n".join([instruction, prefix, filled_template]), context_1, context_2

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
        print(f"Querying entities {i} and {j}")
        if self.queries_cnt < self.max_queries_cnt:
            self.queries_cnt += 1
            sorted_pair_list = sorted([self.ents[i], self.ents[j]])
            sorted_pair = tuple(sorted_pair_list)

            if  sorted_pair in self.gpt3_pairwise_labels:
                return self.filter_high_entropy_predictions(self.gpt3_pairwise_labels[sorted_pair])

            prompt, context1, context2 = self.construct_pairwise_oracle_prompt(i, j)
            print("PROMPT:\n" + prompt)

            pair_labels_not_none = []

            failure = True
            num_retries = 0
            while failure and num_retries < self.NUM_RETRIES:
                cache_row = None
                try:
                    start = time.perf_counter()
                    print(f"Querying {self.ents[i]} and {self.ents[j]}...")
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "user", "content": prompt},
                        ],
                        temperature=1.0,
                        max_tokens=1,
                        n=self.num_predictions,
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

                    print(f"labels:\n{pair_labels}\n\n")
                    pair_labels_not_none = [x for x in pair_labels if x is not None]
                    if len(pair_labels_not_none) <= self.num_predictions / 2:
                        time.sleep(0.8)
                    else:
                        cache_row = {"entity1": self.ents[i],
                                     "entity2": self.ents[j],
                                     "labels": pair_labels_not_none,
                                     "p_true": round(sum(pair_labels_not_none) / len(pair_labels_not_none), 4),
                                     "context1": context1,
                                     "context2": context2
                                     }
                        self.cache_writer.write(cache_row)
                        self.gpt3_pairwise_labels[sorted_pair] = pair_labels_not_none
                        failure = False


                    num_retries += 1
                    end = time.perf_counter()
                    if end - start < 1:
                        time.sleep(1 - (end - start))
                except Exception as e:
                    print(e)
                    time.sleep(3)

            if failure:
                return None
            else:
                return self.filter_high_entropy_predictions(pair_labels_not_none)
        else:
            breakpoint()
            raise MaximumQueriesExceeded
