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

class GPTExpansionClustering(KMeans):
    def __init__(self, n_clusters=3, side_information=None, read_only=True):
        self.n_clusters = n_clusters
        self.side_information = side_information
        self.cache_dir = "/projects/ogma1/vijayv/okb-canonicalization/clustering/file/gpt3_cache"
        self.cache_file = os.path.join(self.cache_dir, "gpt_paraphrase_cache.jsonl")
        if os.path.exists(self.cache_file):
            self.cache_rows = list(jsonlines.open(self.cache_file))
        else:
            self.cache_rows = []
        if not read_only:
            self.cache_writer = jsonlines.open(self.cache_file, mode='a')
        else:
            self.cache_writer = jsonlines.open(self.cache_file, mode='r')
        self.NUM_RETRIES = 1
        self.read_only = read_only

        self.sentence_unprocessing_mapping_file = os.path.join(self.cache_dir, "sentence_unprocessing_map.json")

    def process_sentence_punctuation(self, sentences):
        processed_sentence_set = []
        for s in sentences:
            processed_sentence_set.append(s.replace("-LRB-", "(").replace("-RRB-", ")"))
        return processed_sentence_set


    def gpt3_template(self, entity_name, context_sentences):
        prefix = f"""I am trying to cluster entity strings on Wikipedia according to the Wikipedia article title they refer to. To help me with this, for a given entity name, please provide me with a comprehensive set of alternative names that could refer to the same entity. Entities may be weirdly truncated or ambiguous - e.g. "Wind" may refer to the band "Earth, Wind, and Fire" or to "rescue service". For each entity, I will provide you with a sentence where this entity is used to help you understand what this entity refers to. Generate a comprehensive set of alternate entity names as a JSON-formatted list.

Entity: "fictional character"
Context Sentence: "Jenna Marshall is a fictional character created by Sara Shepard for the `` Pretty Little Liars '' book series , and later developed for the Freeform television series adaptation by I. Marlene King and portrayed by Tammin Sursok ."
Alternate Entity Names: ["fictional characters", "characters", "character"]

Entity: "Catholicism"
Context Sentence: "At home , significantly more electorate residents spoke Italian , Cantonese , Mandarin and Greek at home , and whilst the top three religions (Catholicism , no religion and Anglicanism) differed little from other parts of Perth , Buddhism and Eastern Orthodox adherents outnumbered those of the Uniting Church ."
Alternate Entity Names: ["Catholic Church", "Roman Catholic", "Catholic"]

Entity: "Wind"
Context Sentence: "Illinois musicians with a # 1 Billboard Hot 100 hit include artists from the 1950s : Sam Cooke (d. 1964) ; from the 1960s : The Buckinghams ; from the 1970s : Earth , Wind & Fire , The Chi-Lites , The Staple Singers , Minnie Riperton , Styx ; from the 1980s : Chicago , Cheap Trick , REO Speedwagon , Survivor , Richard Marx ; from the 1990s : R. Kelly ; from the 2000s : Kanye West , Twista , Plain White T 's ."
Alternate Entity Names: ["Earth & Fire", "Earth", "Wind & Fire"]

Entity: "Elizabeth"
Context Sentence: "They had 11 children : Their eldest son , Claude Bowes-Lyon , 14th Earl of Strathmore and Kinghorne , was the father of Queen Elizabeth , Queen consort of King George VI and through her , maternal grandfather of Queen Elizabeth II ; and their fifth son , Patrick Bowes-Lyon (5 March 1863 -- 5 October 1946) , was a Major in the British Army ."
Alternate Entity Names: ["Elizabeth II", "HM"]"""

        filled_template = f"""Entity: "{entity_name}"
Context Sentence: "{context_sentences[0]}"
Alternate Entity Names: """
        return prefix, filled_template

    def fit(self, X, y=None, ml=[], cl=[]):

        side_info = self.side_information.side_info
        ents = []
        sentence_idxs = []
        sentences = []
        selected_sentences = []

        sentence_unprocessing_mapping = json.load(open(self.sentence_unprocessing_mapping_file))

        for i in range(len(X)):
            ents.append(side_info.id2ent[i])
            sentence_idxs.append(side_info.ent_id2sentence_list[i])
            unprocessed_sentences = [sentence_unprocessing_mapping[side_info.sentence_List[j]] for j in sentence_idxs[-1]]
            sentences.append(self.process_sentence_punctuation(unprocessed_sentences))
            shortest_sentences = sorted(sentences[-1], key=len)[:3]
            # selected_sentences.append(shortest_sentences)

            '''
            Choose longest sentence under 306 characers, as in
            https://github.com/Yang233666/cmvc/blob/6e752b1aa5db7ff99eb2fa73476e392a00b0b89a/Context_view.py#L98
            '''
            longest_sentences = sorted([s for s in sentences[-1] if len(s) < 599], key=len, reverse=True)
            if len(longest_sentences) == 0:
                breakpoint()
            selected_sentences.append([longest_sentences[0]])

        entity_expansion_mapping = {}
        for row in self.cache_rows:
            entity_expansion_mapping[row["entity"]] = row["expansion"]

        for ent_idx, entity in tqdm(enumerate(ents)):
            if entity in entity_expansion_mapping:
                entity_expansions = entity_expansion_mapping[entity]
            else:
                if self.read_only:
                    continue
                prefix, filled_template = self.gpt3_template(entity, selected_sentences[side_info.ent2id[entity]])

                failure = True
                num_retries = 0
                while failure and num_retries < self.NUM_RETRIES:
                    cache_row = None
                    try:
                        start = time.perf_counter()
                        response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": f"{prefix}"},
                                {"role": "user", "content": filled_template},
                            ],
                        )
                        message = json.loads(str(response.choices[0]))["message"]["content"]
                        try:
                            entity_expansions = json.loads(message)
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


        entity_expansion_mapping = {}
        for row in self.cache_rows:
            entity_lowercase = row["entity"].lower()
            expanded_lowercase = [r.lower() for r in row["expansion"]]
            entity_expansion_mapping[row["entity"]] = set([entity_lowercase] + expanded_lowercase)

        clusters = [[ents[0]]]
        for ent in ents[1:]:
            if ent not in entity_expansion_mapping:
                entity_expansion_mapping[ent] = set([ent])
            any_cluster_match = None
            for clust_idx, cluster in enumerate(clusters):
                cluster_match = False
                for candidate_ent in cluster:
                    if len(entity_expansion_mapping[ent].intersection(entity_expansion_mapping[candidate_ent])) > 0:
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


        breakpoint()

        return self
