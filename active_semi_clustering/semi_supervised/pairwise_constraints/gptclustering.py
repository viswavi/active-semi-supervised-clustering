from collections import Counter, defaultdict
import json
import jsonlines
import numpy as np
import openai
import os
from sklearn.preprocessing import normalize
import time
from tqdm import tqdm


from active_semi_clustering.exceptions import EmptyClustersException
from .constraints import preprocess_constraints
from active_semi_clustering.semi_supervised.labeled_data.kmeans import KMeans
from active_semi_clustering.semi_supervised.pairwise_constraints.pckmeans import PCKMeans
from InstructorEmbedding import INSTRUCTOR
from sentence_transformers import SentenceTransformer

import sys
sys.path.append("cmvc")
from cmvc.helper import invertDic
from cmvc.metrics import pairwiseMetric, calcF1
from cmvc.test_performance import cluster_test

class GPTExpansionClustering(KMeans):
    def __init__(self, X, dataset_name, labels, split=None, n_clusters=3, side_information=None, read_only=False, instruction_only=False, demonstration_only=False, cache_file_name="gpt_paraphrase_cache.jsonl"):
        self.X = X
        self.dataset_name = dataset_name
        self.labels = labels
        self.n_clusters = n_clusters
        self.side_information = side_information
        self.cache_dir = "/projects/ogma1/vijayv/okb-canonicalization/clustering/file/gpt3_cache"
        cache_file = os.path.join(self.cache_dir, cache_file_name)
        self.instruction_only = instruction_only
        self.demonstration_only = demonstration_only
        if instruction_only:
            filename_components = cache_file.split("_cache.jsonl")
            cache_file = filename_components[0] + f"_instruction_only" + "_cache.jsonl"
        elif demonstration_only:
            filename_components = cache_file.split("_cache.jsonl")
            cache_file = filename_components[0] + f"_demonstration_only" + "_cache.jsonl"
        if os.path.exists(cache_file):
            self.cache_rows = list(jsonlines.open(cache_file))
        else:
            self.cache_rows = []
        if not read_only:
            self.cache_writer = jsonlines.open(cache_file, mode='a', flush=True)
        else:
            self.cache_writer = jsonlines.open(cache_file, mode='r')
        self.NUM_RETRIES = 1
        self.read_only = read_only

        split_str = f"_{split}" if split else ""
        self.sentence_unprocessing_mapping_file = os.path.join(self.cache_dir, f"{dataset_name}{split_str}_sentence_unprocessing_map.json")

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
        elif self.dataset_name == "tweet":
            instruction = f"I am trying to cluster tweets based on whether " + \
            "they discuss the same topic. To do this, given a (stopword-removed) tweet, please " + \
            "provide a comprehensive set of keywords or keyphrases that could describe this tweet's " + \
            "topic. These keywords should be distinct from those that might describe tweets with different " + \
            "topics. Since the tweets already look like keywords, feel free to include keywords not listed " + \
            "in the tweet, and don't feel like you need to include very many of the original words from the " + \
            "tweet. Generate a comprehensive set of keyphrases as a JSON-formatted list."
        elif self.dataset_name == "clinc":
            instruction = f"I am trying to cluster task-oriented dialog system queries based on " + \
            "whether they express the same general user intent. To help me with this, " + \
            "for a given user query, provide a comprehensive set of keywords that could describe " + \
            "this query's intent. These keywords should be distinct from those that might describe " + \
            "queries with different intents. Generate the set of keyphrases as a JSON-formatted list."
        elif self.dataset_name == "bank77":
            instruction = f"I am trying to cluster queries for a online banking system based " + \
            "on whether they express the same general user intent. To help me with this, " + \
            "for a given banking query, provide a comprehensive set of keywords that could describe " + \
            "this query's intent. These keywords should be distinct from those that might describe " + \
            "banking-related queries with different intents. Generate " + \
            "the set of keyphrases as a JSON-formatted list."
        else:
            raise NotImplementedError
        return instruction

    @staticmethod
    def construct_context_sentences(entity_idx, selected_sentences):
        context_labels = ["1", "2", "3", "4"]
        return "\n".join([context_labels[ind] + ") " + '"' + sent + '"' for ind, sent in enumerate(selected_sentences[entity_idx])])

    def create_template_block_entity_canonicalization(self, entity_name, selected_sentences, complete_block=True):
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

    def create_template_block_text_clustering(self, sentence, text_type, keywords, complete_block=True):
        if complete_block:
            keywords_str = json.dumps(keywords)
        else:
            keywords_str = ""
        filled_template = f"""{text_type}: "{sentence}"

Keywords: {keywords_str}"""
        return filled_template

    def get_gpt3_prefix_entity_canonicalization(self, demonstration_entities, selected_sentences, instruction_only=False, demonstration_only=False):
        instruction = self.get_instruction()
        prefix = instruction
        demonstration_blocks = [self.create_template_block_entity_canonicalization(entity_name, selected_sentences, complete_block=True) for entity_name in demonstration_entities]
        prefix = instruction + "\n\n" + "\n\n".join(demonstration_blocks)
        if instruction_only:
            return instruction
        if demonstration_only:
            return "\n\n".join(demonstration_blocks)
        return prefix

    def get_gpt3_prefix_text_clustering(self, sentences, text_type, all_keywords, instruction_only=False, demonstration_only=False):
        instruction = self.get_instruction()
        prefix = instruction
        demonstration_blocks = [self.create_template_block_text_clustering(sentence, text_type, keywords, complete_block=True) for sentence, keywords in zip(sentences, all_keywords)]
        prefix = instruction + "\n\n" + "\n\n".join(demonstration_blocks)
        if instruction_only:
            return instruction
        if demonstration_only:
            return "\n\n".join(demonstration_blocks)
        return prefix

    def expand_entity(self, entity_name):
        entity_idx = self.side_information.side_info.ent2id[entity_name]
        gt_cluster = self.labels[entity_idx]
        gt_coclustered_entity_idxs = [i for i, l in enumerate(self.labels) if l == gt_cluster and i != entity_idx]
        entity_expansions = [self.side_information.side_info.id2ent[entity_idx] for entity_idx in gt_coclustered_entity_idxs]
        return entity_expansions

    def construct_gpt3_template(self, entity_name, selected_sentences, gt_keywords=None, instruction_only=False, demonstration_only=False):
        if self.dataset_name == "OPIEC59k":
            prompt_prefix = self.get_gpt3_prefix_entity_canonicalization(["fictional character", "Catholicism", "Wind", "Elizabeth"], selected_sentences, instruction_only=instruction_only, demonstration_only=demonstration_only)
        elif self.dataset_name == "reverb45k":
            prompt_prefix = self.get_gpt3_prefix_entity_canonicalization(["Hank Aaron", "Apple", "Jason", "Insomniac Games"], selected_sentences, instruction_only=instruction_only, demonstration_only=demonstration_only)
        else:
            if self.dataset_name == "tweet":
                text_type = "Tweet"
                sentences = [
                    "brain fluid buildup delay giffords rehab",
                    "trailer talk week movie rite mechanic week opportunity",
                    "gbagbo camp futile cut ivory coast economy",
                    "chicken cavatelli soup"
                ]
                keywords = [
                    ["gabrielle giffords", "giffords recovery"],
                    ["movies", "in theaters", "trailer talk"],
                    ["gbagbo", "ivory coast"],
                    ["cooking", "tasty recipes"]
                ]
                prompt_prefix = self.get_gpt3_prefix_text_clustering(sentences, text_type, keywords, instruction_only=instruction_only, demonstration_only=demonstration_only)      
            elif self.dataset_name == "clinc":
                text_type = "Query"
                sentences = [
                    "how would you say fly in italian",
                    "what does assiduous mean",
                    "find my cellphone for me!"
                ]
                keywords = [
                    ["translation", "translate"],
                    ["definition", "define"],
                    ["location", "find", "locate", "tracking", "track"],
                ]
                prompt_prefix = self.get_gpt3_prefix_text_clustering(sentences, text_type, keywords, instruction_only=instruction_only, demonstration_only=demonstration_only)      
            elif self.dataset_name == "bank77":
                text_type = "Query"
                sentences = [
                    "How do I locate my card?",
                    "Whats the delivery time to the United States?",
                    "Can you cancel my purchase?",
                    "Why don't I have my transfer?"]
                keywords = [
                    ["card status", "status update", "card location"],
                    ["delivery time", "ETA", "card delivery"],
                    ["cancel purchase", "refund"],
                    ["transfer", "transfer failed"]
                ]
                prompt_prefix = self.get_gpt3_prefix_text_clustering(sentences, text_type, keywords, instruction_only=instruction_only, demonstration_only=demonstration_only)     
            else:
                raise NotImplementedError

        if self.dataset_name == "OPIEC59k" or self.dataset_name == "reverb45k":
            completion_block = self.create_template_block_entity_canonicalization(entity_name, selected_sentences, complete_block=False)
        else:
            sentence = entity_name
            if self.dataset_name == "tweet":
                text_type = "Tweet"      
            elif self.dataset_name == "clinc":
                text_type = "Query"
            elif self.dataset_name == "bank77":
                text_type = "Query"
            completion_block = self.create_template_block_text_clustering(sentence, text_type, None, complete_block=False)
        return f"{prompt_prefix}\n\n{completion_block}"


    def evaluate(self):
        ave_prec, ave_recall, ave_f1, macro_prec, micro_prec, pair_prec, macro_recall, micro_recall, pair_recall, macro_f1, micro_f1, pairwise_f1, model_clusters, model_Singletons, gold_clusters, gold_Singletons  = cluster_test(self.side_information.p, self.side_information.side_info, self.labels_, self.side_information.true_ent2clust, self.side_information.true_clust2ent)
        metrics_dict = {"ave_prec": ave_prec,
                        "ave_recall": ave_recall,
                        "ave_f1": ave_f1,
                        "macro_prec": macro_prec,
                        "micro_prec": micro_prec,
                        "pair_prec": pair_prec,
                        "macro_recall": macro_recall,
                        "micro_recall": micro_recall,
                        "pair_recall": pair_recall,
                        "macro_f1": macro_f1,
                        "micro_f1": micro_f1,
                        "pairwise_f1": pairwise_f1}
        return ave_f1, macro_f1, micro_f1, pairwise_f1, metrics_dict

    def fit(self, X, y=None, ml=[], cl=[]):

        ents = []
        sentence_idxs = []
        sentences = []
        selected_sentences = []

        if self.dataset_name == "OPIEC59k" or self.dataset_name == "reverb45k":
            sentence_unprocessing_mapping = json.load(open(self.sentence_unprocessing_mapping_file))
        else:
            sentence_unprocessing_mapping = None

        for i in range(len(X)):
            if self.dataset_name == "OPIEC59k" or self.dataset_name == "reverb45k":
                ents.append(self.side_information.side_info.id2ent[i])
                entity_sentence_idxs = self.side_information.side_info.ent_id2sentence_list[i]
                unprocessed_sentences = [sentence_unprocessing_mapping[self.side_information.side_info.sentence_List[j]] for j in entity_sentence_idxs]
                entity_sentences = self.process_sentence_punctuation(unprocessed_sentences)
                entity_sentences_dedup = list(set(entity_sentences))

                '''
                Choose longest sentence under 306 characers, as in
                https://github.com/Yang233666/cmvc/blob/6e752b1aa5db7ff99eb2fa73476e392a00b0b89a/Context_view.py#L98
                '''
                longest_sentences = sorted([s for s in entity_sentences_dedup if len(s) < 599], key=len)
                selected_sentences.append(longest_sentences[:3])
                sentences.append(longest_sentences[0])
            else:
                ents.append(self.side_information[i])
                selected_sentences.append(self.side_information[i])
                sentences.append(self.side_information[i])



        entity_expansion_mapping = {}
        for row in self.cache_rows:
            entity_expansion_mapping[row["entity"]] = row["expansion"]

        for ent_idx, entity in tqdm(enumerate(ents)):
            if entity not in entity_expansion_mapping:
                if self.read_only:
                    continue
                template_to_fill = self.construct_gpt3_template(entity, selected_sentences, instruction_only=self.instruction_only, demonstration_only=self.demonstration_only)
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
                        if message.startswith("Keywords:"):
                            message = message[len("Keywords:"):].strip()
                        try:
                            entity_expansions = json.loads(message)
                            print(message)
                            if not isinstance(entity_expansions, list) or not isinstance(entity_expansions[0], str):
                                failure = True
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
        all_keywords = set()
        for ent, expansions in entity_expansion_mapping.items():
            _ = """
            entity_lowercase = ent.lower().split()
            expanded_lowercase = [r.lower() for r in expansions]
            expanded_lowercase = [e for r in expansions for e in r.lower().split()]
            # entity_expansions_lowercase[ent] = set(expanded_lowercase)
            entity_expansions_lowercase[ent] = set(entity_lowercase + expanded_lowercase)
            """
            entity_lowercase = ent.lower()
            try:
                expanded_lowercase = [r.lower() for r in expansions]
            except:
                breakpoint()
            # entity_expansions_lowercase[ent] = set(expanded_lowercase)
            entity_expansions_lowercase[ent] = set([entity_lowercase] + expanded_lowercase)
            all_keywords.update(entity_expansions_lowercase[ent])

        for ent in ents:
            if ent not in entity_expansions_lowercase:
                # entity_expansions_lowercase[ent] = set(ent.lower().split())
                entity_expansions_lowercase[ent] = set([ent.lower()])

        _ = """
        all_keywords = []
        for ent in ents:
            all_keywords.extend(list(entity_expansions_lowercase[ent]))

        keyword_counts = Counter(all_keywords)
        most_common_keywords = [k for k,v in keyword_counts.items() if v > (len(ents) // self.n_clusters)]
        
        for ent in entity_expansions_lowercase:
            for common_keyword in most_common_keywords:
                if common_keyword in entity_expansions_lowercase[ent]:
                    entity_expansions_lowercase[ent].remove(common_keyword)
        """

        ents_sorted = sorted(ents, key=lambda e: len(entity_expansions_lowercase[e]), reverse=True)

        clusters = [[ents_sorted[0]]]
        cluster_keywords = [set(entity_expansions_lowercase[ents_sorted[0]])]
        for ent in ents_sorted[1:]:
            any_cluster_match = None
            for clust_idx, cluster in enumerate(clusters):
                cluster_match = False
                for candidate_ent in cluster:
                    # if len(entity_expansions_lowercase[ent].intersection(entity_expansions_lowercase[candidate_ent])) > 0:
                    jaccard_similarity = len(entity_expansions_lowercase[ent].intersection(entity_expansions_lowercase[candidate_ent])) / len(entity_expansions_lowercase[ent].union(entity_expansions_lowercase[candidate_ent]))
                    if jaccard_similarity > 0:
                        cluster_match = True
                        break
                if cluster_match:
                    any_cluster_match = clust_idx
                    break
            if any_cluster_match is not None:
                clusters[any_cluster_match].append(ent)
                cluster_keywords[any_cluster_match].update(entity_expansions_lowercase[ent])
            else:
                clusters.append([ent])
                cluster_keywords.append(set(entity_expansions_lowercase[ent]))

        clusters_and_keywords = []
        for (single_cluster_ents, single_cluster_keywords) in zip(clusters, cluster_keywords):
            clusters_and_keywords.append((list(single_cluster_ents), list(single_cluster_keywords)))

        clusters_and_keywords = sorted(clusters_and_keywords, key=lambda k: len(k[0]))

        # post process clusters

        # compute labels
        ent_to_cluster_idx = {}
        for clust_idx, cluster in enumerate(clusters):
            for ent in cluster:
                ent_to_cluster_idx[ent] = clust_idx
        if isinstance(self.side_information, list):
            self.labels_ = [ent_to_cluster_idx[self.side_information[idx]] for idx in range(len(X))]
        else:
            self.labels_ = [ent_to_cluster_idx[self.side_information.side_info.id2ent[idx]] for idx in range(len(X))]



        all_expansions = []
        for ent in ents:
            if self.dataset_name == "OPIEC59k" or self.dataset_name == "reverb45k":
                ent_expansions = [ent]
            else:
                ent_expansions = []
            if ent in entity_expansion_mapping:
                ent_expansions.extend(entity_expansion_mapping[ent])
            try:
                all_expansions.append(", ".join(ent_expansions))
            except:
                breakpoint()

        _ = """
        if self.dataset_name == "OPIEC59k" or self.dataset_name == "reverb45k"  or self.dataset_name == "tweet" or self.dataset_name == "bank77":
            model = SentenceTransformer('sentence-transformers/distilbert-base-nli-stsb-mean-tokens')
            embeddings = model.encode(list(all_keywords))
        elif self.dataset_name == "bank77":
            model = INSTRUCTOR('hkunlp/instructor-large')
            prompt = "Represent the bank purpose for intent classification: "
            embeddings = model.encode([[prompt, text] for text in all_keywords])

        keyword_to_embedding = dict(zip(list(all_keywords), embeddings))
        ent_multi_keyword_embeddings = [[keyword_to_embedding[e] for e in entity_expansions_lowercase[ent]] for ent in ents]
        ent_keyword_embeddings = np.stack([np.mean(vecs, axis=0) for vecs in ent_multi_keyword_embeddings])
        """


        if self.dataset_name == "OPIEC59k" or self.dataset_name == "reverb45k"  or self.dataset_name == "tweet":
            model = SentenceTransformer('sentence-transformers/distilbert-base-nli-stsb-mean-tokens')
            ent_embeddings = model.encode(ents)
            expansion_embeddings = model.encode(all_expansions)
        elif self.dataset_name == "bank77":
            model = INSTRUCTOR('hkunlp/instructor-large')
            prompt = "Represent the bank purpose for intent classification: "
            ent_embeddings = model.encode([[prompt, ent] for ent in ents])
            expansion_embeddings = model.encode([[prompt, text] for text in all_expansions])
        elif self.dataset_name == "clinc":
            model = INSTRUCTOR('hkunlp/instructor-large')
            prompt = "Represent keyphrases for topic classification: "
            ent_embeddings = model.encode([[prompt, ent] for ent in ents])
            expansion_embeddings = model.encode([[prompt, text] for text in all_expansions])
        else:
            raise ValueError(f"Dataset {self.dataset_name} not found")

        a_vectors = normalize(self.X, axis=1, norm="l2")
        b_vectors = normalize(expansion_embeddings, axis=1, norm="l2")
        embeddings = np.concatenate([a_vectors, b_vectors], axis=1)

        kmeans = KMeans(self.n_clusters, max_iter=100, init="k-means++", normalize_vectors=True, split_normalization=True, split_point=np.shape(self.X)[1])
        kmeans.fit(embeddings)
        self.labels_ = [int(l) for l in kmeans.labels_]
        return self
