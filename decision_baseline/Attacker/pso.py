import numpy as np
import copy
from tqdm import tqdm
valid_pos_list = ['NOUN', 'VERB', 'ADV', 'ADJ']
class AttackPSO():
    def __init__(self, model, word_candidate, word_dict, max_iters=100, pop_size=60):
        self.model = model
        self.word_candidate = word_candidate
        self.word_dict = word_dict
        self.inv_word_dict = {idx: w for w, idx in word_dict.items()}
        self.max_iters = max_iters
        self.pop_size = pop_size

        self.temp = 0.3


    def mutate(self, orig_text, select_prob, w_list):
        x_len = orig_text.shape[0]
        rand_idx = np.random.choice(x_len, 1, p=select_prob)[0]
        replace_text = self.do_replace(orig_text, rand_idx, w_list[rand_idx])
        return replace_text


    def generate_population(self, orig_text, neighbor_list, pop_size, text_len,neighbor_length):
        w_list, prob_list = self.gen_h_score(text_len, neighbor_length, neighbor_list, orig_text)
        x_len = orig_text.shape[0]
        prob_len = prob_list.shape[0]
        if x_len != prob_len:
            return None
        all_generated_orig_text = [self.mutate(orig_text, prob_list, w_list) for _ in range(pop_size)]
        return all_generated_orig_text





    def norm(self, prob_list):
        new_prob_list = []
        for prob in prob_list:
            if prob <= 0:
                new_prob_list.append(0)
            else:
                new_prob_list.append(prob)
        total = np.sum(new_prob_list)

        if total == 0:
            new_prob_list = (np.ones_like(new_prob_list) / len(new_prob_list)).tolist()
        else:
            new_prob_list = [prob / total for prob in new_prob_list]

        return new_prob_list

    def gen_h_score(self, text_len, neighbor_length, neighbor_list, orig_text):
        w_list = []
        prob_list = []
        for i in range(text_len):
            if neighbor_length[i] == 0:
                w_list.append(orig_text[i])
                prob_list.append(0)
                continue
            w, p = self.gen_most_changes(i, orig_text, neighbor_list[i])
            w_list.append(w)
            prob_list.append(p)

        h_score = self.norm(prob_list)

        # assert len(h_score) == len(w_list) == orig_text.shape[0]
        h_score = np.array(h_score)

        return w_list, h_score

    def gen_most_changes(self, idx, orig_text, single_neighbor_list):
        new_x_list = [self.do_replace(orig_text, idx, w) if orig_text[idx] != w else orig_text for w in single_neighbor_list]

        x_text_list = [' '.join([self.inv_word_dict[ids_list[idx]] for idx in range(ids_list.size)]) for ids_list in new_x_list]
        if len(x_text_list) == 1:
            flags, scores = self.model.predict(x_text_list[0], batch=False)
        else:
            flags, scores = self.model.predict(x_text_list, batch=True)
        _, orig_score = self.model.predict(' '.join([self.inv_word_dict[orig_text[idx]] for idx in range(orig_text.size)]), batch=False)
        # print(scores, orig_score)
        new_x_scores = np.array(scores) - orig_score
        max_delta_score = np.max(new_x_scores)
        target_word = new_x_list[np.argsort(new_x_scores)[-1]][idx]
        return target_word, max_delta_score,






    def do_replace(self, orig_text, idx, new_word):
        x_new = orig_text.copy()
        x_new[idx] = new_word
        return x_new




    def equal(self,a,b):
        return -3 if a == b else 3

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


    def turn(self, elites, orig, turn_prob, x_len):
        new_text = copy.deepcopy(orig)
        for i in range(x_len):
            if np.random.uniform() < turn_prob[i]:
                new_text[i] = elites[i]
        return new_text

    def count_change_ratio(self, text1,text2, x_len):
        change_ratio = float(np.sum(text1 != text2)) / float(x_len)
        return change_ratio



    def attack(self, orig_text, pos_list):
        x_adv = orig_text.copy()
        x_len = int(np.sum(np.sign(x_adv)))
        if x_len != len(pos_list):
            # print('exception happen')
            return 0
        print('len measure: ', x_len, ' ', len(pos_list))
        neighbor_list = []
        for i in range(x_len):
            word = x_adv[i]
            assert word in range(0, len(self.word_dict))
            word_pos = pos_list[i]
            if word_pos not in valid_pos_list:
                neighbor_list.append([])
                continue
            candidates_list = self.word_candidate[word]
            neighbor_list.append(candidates_list[word_pos])
        neighbor_length = [len(neighbor) for neighbor in neighbor_list]
        if np.sum(neighbor_length) == 0:
            return None
        print('neighbor length: ', neighbor_length)
        try:
            all_generated_orig_text = self.generate_population(orig_text, neighbor_list, self.pop_size, x_len,neighbor_length)
        except ValueError:
            return 0
        if all_generated_orig_text is None:
            return 0

        particles_elites = copy.deepcopy(all_generated_orig_text)  # [np.array([1,2,3]), np.array([1,2,3]) ]
        input_particles = [' '.join([self.inv_word_dict[ids_list[idx]] for idx in range(ids_list.size)]) for ids_list in particles_elites]
        flags, scores = self.model.predict(input_particles, batch=True)
        part_elites_scores = scores

        max_score = np.max(scores)
        particles_rank = np.argsort(scores)[::-1] # from bigger to smaller

        top_attack = particles_rank[0]
        all_elite_particle = all_generated_orig_text[top_attack]

        if flags[top_attack] == 1:
            return all_elite_particle

        Omega_1 = 0.8
        Omega_2 = 0.2
        C1_origin = 0.8
        C2_origin = 0.2

        V = [np.random.uniform(-3, 3) for _ in range(self.pop_size)]
        V_P = [[V[t] for _ in range(x_len)] for t in range(self.pop_size)]

        for i in range(self.max_iters):
            Omega = (Omega_1 - Omega_2) * (self.max_iters - i) / self.max_iters + Omega_2

            C1 = C1_origin - i / self.max_iters * (C1_origin - C2_origin)
            C2 = C2_origin + i / self.max_iters * (C1_origin - C2_origin)


            for id in range(self.pop_size):
                for dim in range(x_len):
                    V_P[id][dim] = Omega * V_P[id][dim] + (1 - Omega) * (
                                 self.equal(all_generated_orig_text[id][dim], particles_elites[id][dim])
                                 + self.equal(all_generated_orig_text[id][dim], all_elite_particle[dim]))
                turn_prob = [self.sigmoid(V_P[id][d]) for d in range(x_len)]

                P1 = C1
                P2 = C2

                if np.random.uniform() < P1:
                    all_generated_orig_text[id] = self.turn(particles_elites[id], all_generated_orig_text[id],
                                                            turn_prob, x_len)
                if np.random.uniform() < P2:
                    all_generated_orig_text[id] = self.turn(all_elite_particle, all_generated_orig_text[id],
                                                            turn_prob, x_len)

            input_origin_text = [' '.join([self.inv_word_dict[ids_list[idx]] for idx in range(ids_list.size)]) for ids_list in all_generated_orig_text]
            flags, scores = self.model.predict(input_origin_text, batch=True)

            pop_ranks = np.argsort(scores)[::-1]
            top_attack = pop_ranks[0]

            print('******', i, ' ********', 'before mutation : ', scores[top_attack])

            if flags[top_attack] == 1:
                return all_generated_orig_text[top_attack]

            new_generated_text = []
            for text in all_generated_orig_text:
                change_ratio = self.count_change_ratio(text, orig_text, x_len)
                p_change = 1 - 2 * change_ratio
                if np.random.uniform() < p_change:
                    w_list, h_score = self.gen_h_score(x_len, neighbor_length, neighbor_list, text)
                    mutated_text = self.mutate(text, h_score, w_list)
                    new_generated_text.append(mutated_text)
                else:
                    new_generated_text.append(text)


            all_generated_orig_text = new_generated_text

            input_generated_text = [' '.join([self.inv_word_dict[ids_list[idx]] for idx in range(ids_list.size)]) for
                                    ids_list in all_generated_orig_text]
            flags, scores = self.model.predict(input_generated_text, batch=True)
            pop_ranks = np.argsort(scores)[::-1]
            top_attack = pop_ranks[0]

            print('******', i, ' ********', 'after mutation : ', scores[top_attack])

            if flags[top_attack] == 1:
                return all_generated_orig_text[top_attack]

            for k in range(self.pop_size):
                if scores[k] > part_elites_scores[k]:
                    part_elites_scores[k] = scores[k]
                    particles_elites[k] = all_generated_orig_text[k]
            elite = all_generated_orig_text[top_attack]
            if np.max(scores) > max_score:
                max_score = np.max(scores)
                all_elite_particle = elite

        return None
