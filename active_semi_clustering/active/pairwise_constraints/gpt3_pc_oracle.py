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
            entity_sentences = process_sentence_punctuation(unprocessed_sentences)

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
            self.gpt3_pairwise_labels[tuple(sorted_pair_list)] = row["label"]

    def process_sentence_punctuation(self, sentences):
        processed_sentence_set = []
        for s in sentences:
            processed_sentence_set.append(s.replace("-LRB-", "(").replace("-RRB-", ")"))
        return processed_sentence_set


    def construct_pairwise_oracle_prompt(self, i, j):
        prefix = """I am trying to cluster entity strings on Wikipedia according to the Wikipedia article title they refer to. To help me with this, for a given entity name, please make your best guess as to whether these two objects refer to the same person, location, organization, or object. Entities may be weirdly truncated or ambiguous - e.g. "Wind" may refer to the band "Earth, Wind, and Fire" or to "rescue service". For each entity, I will also provide you with an example sentence from Wikipedia where this entity is referred to. This is just one example where this entity appears and may not be the most representative sentence. Here are a few examples:

1) B.A
Context Sentence: "He matriculated at Jesus College , Oxford on 26 April 1616 , aged 16 , then transferred to Christ 's College , Cambridge ( B.A 1620 , M.A. 1623 , D.D. 1640 ) ."
2) M.D.
Context Sentence: "One study , published in `` The Journal of the American Osteopathic Association Frontier physician Andrew Taylor Still , M.D. , DO , founded the American School of Osteopathy ( now the A.T. Still University-Kirksville ( Mo. ) College of Osteopathic Medicine ) in Kirksville , MO , in 1892 as a radical protest against the turn-of-the-century medical system ."
Given this context, would B.A and M.D. link to the same entity's article on Wikipedia? No

1) B.A
Context Sentence: "He matriculated at Jesus College , Oxford on 26 April 1616 , aged 16 , then transferred to Christ 's College , Cambridge ( B.A 1620 , M.A. 1623 , D.D. 1640 ) ."
2) bachelor
Context Sentence: "After earning a bachelor 's degree from Stanford University and a master 's degree from the Stanford Graduate School of Education , Long was hired as a track coach at Los Altos High School in 1956 , coaching at the school from 1956 -- 1963 and again from 1969 -- 1981 ."
Given this context, would B.A and bachelor link to the same entity's article on Wikipedia? Yes

1) British Government; context sentence: "On 8 September 1939 ,  advised The Football Association ( FA)  clubs could stage friendly matches outside evacuation areas  Liverpool  able  take part   matches , constrained  unavailability  players   services , throughout  war ."
2) government; context sentence: "The amalgamation   two regiments  one   title The Connaught Rangers ,  part   United Kingdom 's reorganization   British Army   Childers Reforms ,  continuation   Cardwell Reforms It  one  eight Irish regiments raised largely  Ireland ,   home depot  Renmore Barracks  Galway ."
Given this context, would British Government and government link to the same entity's article on Wikipedia? Yes

1) Duke of York
Context Sentence: "In supporting John Christian Curwen 's bill for the prevention of the sale of seats , he suggested that the Duke of York and Albany , the late Commander-in-Chief of the Forces , had to some extent corrupted members of parliament ; and in speaking on the budget resolutions of 1808 he declared his belief that the influence of the prerogative had increased ."
2) Frederick
Context Sentence: "These realities could not but influence the nature and direction of Krasicki 's subsequent literary productions , perhaps nowhere more so than in the `` Fables and Parables Soon after the First Partition , Krasicki officiated at the 1773 opening of Berlin 's St. Hedwig 's Cathedral , which Frederick had built for Catholic immigrants to Brandenburg and Berlin ."
Given this context, would Duke of York and Frederick link to the same entity's article on Wikipedia? No"""

        filled_template = f"""1) {self.ents[i]}\nContext Sentence: "{self.selected_sentences[i][0]}"\n2) {self.ents[j]}\nContext Sentence: "{self.selected_sentences[j][0]}"\nGiven this context, would {self.ents[i]} and {self.ents[j]} link to the same entity's article on Wikipedia? """

        return prefix + "\n\n" + filled_template

    def query(self, i, j):
        if self.queries_cnt < self.max_queries_cnt:
            self.queries_cnt += 1
            sorted_pair_list = sorted([self.ents[i], self.ents[j]])
            sorted_pair = tuple(sorted_pair_list)
            if  sorted_pair in self.gpt3_pairwise_labels:
                return self.gpt3_pairwise_labels[sorted_pair]

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
                        if message.strip() == "Yes":
                            pair_label = True
                        elif message.strip() == "No":
                            pair_label = False
                        else:
                            pair_label = None
                        cache_row = {"entity1": self.ents[i],
                                     "entity2": self.ents[j],
                                     "label": pair_label}
                        self.cache_writer.write(cache_row)
                        self.gpt3_pairwise_labels[sorted_pair] = pair_label
                        failure = False
                    except:
                        time.sleep(0.8)
                    num_retries += 1
                    end = time.perf_counter()
                    if end - start < 1:
                        time.sleep(1 - (end - start))
                except:
                    time.sleep(3)

            return pair_label
        else:
            raise MaximumQueriesExceeded