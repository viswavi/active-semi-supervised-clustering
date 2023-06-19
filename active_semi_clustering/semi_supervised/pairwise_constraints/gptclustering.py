from collections import defaultdict
import json
import jsonlines
import numpy as np
import openai
import os
import time
from tqdm import tqdm

from active_semi_clustering.exceptions import EmptyClustersException
from .constraints import preprocess_constraints
from active_semi_clustering.semi_supervised.labeled_data.kmeans import KMeans
from active_semi_clustering.semi_supervised.pairwise_constraints.pckmeans import PCKMeans

class GPTExpansionClustering(KMeans):
    def __init__(self, dataset_name, labels, split=None, n_clusters=3, side_information=None, read_only=False, cache_file_name="gpt_paraphrase_cache.jsonl"):
        self.dataset_name = dataset_name
        self.labels = labels
        self.n_clusters = n_clusters
        self.side_information = side_information
        self.cache_dir = "/projects/ogma1/vijayv/okb-canonicalization/clustering/file/gpt3_cache"
        self.cache_file = os.path.join(self.cache_dir, cache_file_name)
        if os.path.exists(self.cache_file):
            self.cache_rows = list(jsonlines.open(self.cache_file))
        else:
            self.cache_rows = []
        if not read_only:
            self.cache_writer = jsonlines.open(self.cache_file, mode='a', flush=True)
        else:
            self.cache_writer = jsonlines.open(self.cache_file, mode='r')
        self.NUM_RETRIES = 1
        self.read_only = read_only

        self.sentence_unprocessing_mapping_file = os.path.join(self.cache_dir, f"{dataset_name}_{split}_sentence_unprocessing_map.json")

    def process_sentence_punctuation(self, sentences):
        processed_sentence_set = []
        for s in sentences:
            processed_sentence_set.append(s.replace("-LRB-", "(").replace("-RRB-", ")"))
        return processed_sentence_set

    def get_instruction(self):
        if self.dataset_name == "OPIEC59k":
            instruction = f"I am trying to cluster entity strings on Wikipedia according " + \
            "to the Wikipedia article title they refer to. To help me with this, " + \
            "for a given entity name, please provide me with a comprehensive set " + \
            "of alternative names that could refer to the same entity. Entities may " + \
            """be weirdly truncated or ambiguous - e.g. "Wind" may refer to the """ + \
            """band "Earth, Wind, and Fire" or to "rescue service". For each """ + \
            "entity, I will provide you with a sentence where this entity is used " + \
            "to help you understand what this entity refers to. Generate a " + \
            "comprehensive set of alternate entity names as a JSON-formatted list."
        elif self.dataset_name == "reverb45k":
            instruction = f"I am trying to cluster entity strings from the Internet " + \
            "according to the Freebase knowledge graph entity that they " + \
            "refer to. To help me with this, for a given entity name, " + \
            "please provide me with a comprehensive set of alternative " + \
            "names that could refer to the same entity. For each entity, " + \
            "I will provide you with a sentence where this entity is used " + \
            "to help you understand what this entity refers to. Some entities have no " + \
            "alternate names, while others will have alternate names not directly " + \
            "mentioned in the context sentences. Generate a comprehensive set of " + \
            "alternate entity names as a JSON-formatted list."
        else:
            raise NotImplementedError
        return instruction

    @staticmethod
    def construct_context_sentences(entity_idx, selected_sentences):
        context_labels = ["1", "2", "3", "4"]
        return "\n".join([context_labels[ind] + ") " + '"' + sent + '"' for ind, sent in enumerate(selected_sentences[entity_idx])])

    def create_template_block(self, entity_name, selected_sentences, complete_block=True):
        entity_idx = self.side_information.side_info.ent2id[entity_name]
        if complete_block:
            entity_expansions = self.expand_entity(entity_name)
            if entity_name == "Hank Aaron":
                entity_expansions.append("Aaron")
                entity_expansions = list(set(entity_expansions))
            expansion_text = json.dumps(entity_expansions)
        else:
            expansion_text = ""
        filled_template = f"""Entity: "{entity_name}"

Context Sentences:\n{self.construct_context_sentences(entity_idx, selected_sentences)}"

Alternate Entity Names: {expansion_text}"""
        return filled_template

    def get_gpt3_prefix(self, demonstration_entities, selected_sentences):
        instruction = self.get_instruction()
        prefix = instruction
        # demonstration_blocks = [self.create_template_block(entity_name, selected_sentences, complete_block=True) for entity_name in demonstration_entities]
        # prefix = instruction + "\n\n" + "\n\n".join(demonstration_blocks)
        return prefix

    def expand_entity(self, entity_name):
        entity_idx = self.side_information.side_info.ent2id[entity_name]
        gt_cluster = self.labels[entity_idx]
        gt_coclustered_entity_idxs = [i for i, l in enumerate(self.labels) if l == gt_cluster and i != entity_idx]
        entity_expansions = [self.side_information.side_info.id2ent[entity_idx] for entity_idx in gt_coclustered_entity_idxs]
        return entity_expansions

    def construct_gpt3_template(self, entity_name, selected_sentences):
        if self.dataset_name == "OPIEC59k":
            prompt_prefix = self.get_gpt3_prefix(["fictional character", "Catholicism", "Wind", "Elizabeth"], selected_sentences)
        elif self.dataset_name == "reverb45k":
            prompt_prefix = self.get_gpt3_prefix(["Hank Aaron", "Apple", "Jason", "Insomniac Games"], selected_sentences)
        else:
            raise NotImplementedError
        completion_block = self.create_template_block(entity_name, selected_sentences, complete_block=False)
        return f"{prompt_prefix}\n\n{completion_block}"


    def fit(self, X, y=None, ml=[], cl=[]):

        side_info = self.side_information.side_info
        ents = []
        sentence_idxs = []
        sentences = []
        selected_sentences = []
        ents = []

        sentence_unprocessing_mapping = json.load(open(self.sentence_unprocessing_mapping_file))

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
            selected_sentences.append(longest_sentences[:3])


        entity_expansion_mapping = {}
        for row in self.cache_rows:
            entity_expansion_mapping[row["entity"]] = row["expansion"]

        for ent_idx, entity in tqdm(enumerate(ents)):
            if entity not in entity_expansion_mapping:
                if self.read_only:
                    continue
                template_to_fill = self.construct_gpt3_template(entity, selected_sentences)
                print(f"PROMPT:\n{template_to_fill}")

                failure = True
                num_retries = 0
                while failure and num_retries < self.NUM_RETRIES:
                    cache_row = None
                    try:
                        start = time.perf_counter()
                        response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "user", "content": template_to_fill},
                            ],
                        )
                        message = json.loads(str(response.choices[0]))["message"]["content"]
                        try:
                            entity_expansions = json.loads(message)
                            print(message)
                            entity_expansion_mapping[entity] = entity_expansions
                            cache_row = {"entity": entity, "expansion": entity_expansions}
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
        if not self.read_only:
            self.cache_writer.close()

        """
        cluster_idx_to_entities = defaultdict(list)
        entity_to_cluster_idx = {}
        for i, c in enumerate(y):
            cluster_idx_to_entities[c].append(ents[i])
            entity_to_cluster_idx[ents[i]] = c

        _ = cluster_idx_to_entities[entity_to_cluster_idx["Wind"]]
        """

        entity_expansions_lowercase = {}
        for ent, expansions in entity_expansion_mapping.items():
            entity_lowercase = ent.lower()
            expanded_lowercase = [r.lower() for r in expansions]
            entity_expansions_lowercase[ent] =  set([entity_lowercase] + expanded_lowercase)
        for ent in ents:
            if ent not in entity_expansions_lowercase:
                entity_expansions_lowercase[ent] = set([ent.lower()])

        clusters = [[ents[0]]]
        for ent in ents[1:]:
            any_cluster_match = None
            for clust_idx, cluster in enumerate(clusters):
                cluster_match = False
                for candidate_ent in cluster:
                    if len(entity_expansions_lowercase[ent].intersection(entity_expansions_lowercase[candidate_ent])) > 0:
                        cluster_match = True
                        break
                if cluster_match:
                    any_cluster_match = clust_idx
                    break
            if any_cluster_match is not None:
                clusters[any_cluster_match].append(ent)
            else:
                clusters.append([ent])

        # post process clusters

        # compute labels
        ent_to_cluster_idx = {}
        for clust_idx, cluster in enumerate(clusters):
            for ent in cluster:
                ent_to_cluster_idx[ent] = clust_idx
        self.labels_ = [ent_to_cluster_idx[side_info.id2ent[idx]] for idx in range(len(X))]
        return self
