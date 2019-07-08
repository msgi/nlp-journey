import jieba
import synonyms
import random
from random import shuffle

random.seed(2019)


def load_stopwords(stop_path):
    with open(stop_path, 'r',encoding='utf-8') as f:
        stop_words = f.readlines()
        stop_words = [stop_word.strip() for stop_word in stop_words]
    return stop_words


class EDA:
    def __init__(self, stop_path):
        self.stopwords = load_stopwords(stop_path)

    def synonym_replacement(self, words, n):
        new_words = words.copy()
        random_word_list = list(set([word for word in words if word not in self.stopwords]))
        random.shuffle(random_word_list)
        num_replaced = 0
        for random_word in random_word_list:
            synonyms_ = self.get_synonyms(random_word)
            if len(synonyms_) >= 1:
                synonym = random.choice(synonyms_)
                new_words = [synonym if word == random_word else word for word in new_words]
                num_replaced += 1
            if num_replaced >= n:
                break

        sentence = ' '.join(new_words)
        new_words = sentence.split(' ')

        return new_words

    def get_synonyms(self, word):
        return synonyms.nearby(word)[0]

    def random_insertion(self, words, n):
        new_words = words.copy()
        for _ in range(n):
            self.add_word(new_words)
        return new_words

    def add_word(self, new_words):
        synonyms_ = []
        counter = 0
        while len(synonyms_) < 1:
            random_word = new_words[random.randint(0, len(new_words)-1)]
            synonyms_ = self.get_synonyms(random_word)
            counter += 1
            if counter >= 10:
                return
        random_synonym = random.choice(synonyms_)
        random_idx = random.randint(0, len(new_words)-1)
        new_words.insert(random_idx, random_synonym)

    def random_swap(self, words, n):
        new_words = words.copy()
        for _ in range(n):
            new_words = self.swap_word(new_words)
        return new_words

    def swap_word(self, new_words):
        random_idx_1 = random.randint(0, len(new_words)-1)
        random_idx_2 = random_idx_1
        counter = 0
        while random_idx_2 == random_idx_1:
            random_idx_2 = random.randint(0, len(new_words)-1)
            counter += 1
            if counter > 3:
                return new_words
        new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
        return new_words

    def random_deletion(self, words, p):

        if len(words) == 1:
            return words

        new_words = []
        for word in words:
            r = random.uniform(0, 1)
            if r > p:
                new_words.append(word)

        if len(new_words) == 0:
            rand_int = random.randint(0, len(words)-1)
            return [words[rand_int]]

        return new_words

    def fit_transfrom(self, sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=9):
        seg_list = jieba.cut(sentence)
        seg_list = " ".join(seg_list)
        words = list(seg_list.split())
        num_words = len(words)

        augmented_sentences = []
        num_new_per_technique = int(num_aug/4)+1
        n_sr = max(1, int(alpha_sr * num_words))
        n_ri = max(1, int(alpha_ri * num_words))
        n_rs = max(1, int(alpha_rs * num_words))

        # 同义词替换sr
        for _ in range(num_new_per_technique):
            a_words = self.synonym_replacement(words, n_sr)
            augmented_sentences.append(' '.join(a_words))

        # 随机插入ri
        for _ in range(num_new_per_technique):
            a_words = self.random_insertion(words, n_ri)
            augmented_sentences.append(' '.join(a_words))

        # 随机交换rs
        for _ in range(num_new_per_technique):
            a_words = self.random_swap(words, n_rs)
            augmented_sentences.append(' '.join(a_words))

        # 随机删除rd
        for _ in range(num_new_per_technique):
            a_words = self.random_deletion(words, p_rd)
            augmented_sentences.append(' '.join(a_words))

        shuffle(augmented_sentences)

        if num_aug >= 1:
            augmented_sentences = augmented_sentences[:num_aug]
        else:
            keep_prob = num_aug / len(augmented_sentences)
            augmented_sentences = [
                s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]

        augmented_sentences.append(seg_list)

        return augmented_sentences
