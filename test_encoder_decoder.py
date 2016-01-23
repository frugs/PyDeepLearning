import random
import unittest
import re
import numpy as np
from pydl import Gru, mathutils

class TestEncoderDecoder(unittest.TestCase):
    def testTranslateWordsIntoInitialisms(self):
        text = "Born in Vienna into one of Europe's richest families, he inherited a large fortune " \
               "from his father in 1913. He gave some considerable sums to poor artists. In a period " \
               "of severe personal depression after the first World War, he then gave away his entire " \
               "fortune to his brothers and sisters. Three of his brothers committed suicide, with " \
               "Wittgenstein contemplating it too. He left academia several times—serving as an " \
               "officer on the front line during World War I, where he was decorated a number of times " \
               "for his courage; teaching in schools in remote Austrian villages where he encountered " \
               "controversy for hitting children when they made mistakes in mathematics; and working " \
               "as a hospital porter during World War II in London where he told patients not to take " \
               "the drugs they were prescribed while largely managing to keep secret the fact that he " \
               "was one of the world's most famous philosophers."

        index_to_word = sorted(list(set(text.split(sep=" "))))
        word_to_index = {word:i for i, word in enumerate(index_to_word)}

        index_to_char = sorted(list(set([word[0].upper() for word in index_to_word])))
        char_to_index = {char:i for i, char in enumerate(index_to_char)}

        def vector_from_word(word):
            index = word_to_index[word]
            vec = np.zeros(len(index_to_word))
            vec[index] = 1
            return vec

        def word_from_vector(vector):
            index = vector.argmax()
            if vector[index] < 0.3:
                return "?"
            else:
                return index_to_word[index]

        def vector_from_char(char):
            vec = np.zeros(len(index_to_char))
            upper = char.upper()
            if upper in char_to_index:
                index = char_to_index[upper]
                vec[index] = 1
            return vec

        def char_from_vector(vector):
            index = vector.argmax()
            if vector[index] < -0.3:
                return " "
            elif vector[index] < 0.3:
                return "?"
            else:
                return index_to_char[index]

        max_seq_size = 5

        training_set = []
        for _ in range(500):
            seq_size = random.randint(1, max_seq_size)
            word_indices = [random.randrange(0, len(index_to_word)) for _ in range(seq_size)]
            words = [index_to_word[index] for index in word_indices]

            initials = [word[0].upper() for word in words]

            training_set.append(([vector_from_word(word) for word in words],
                                 [vector_from_char(char) for char in initials]))

        encoder_hidden_state_size = len(index_to_char)
        encoder = Gru(len(index_to_word), encoder_hidden_state_size)
        decoder = Gru(len(index_to_char) + encoder_hidden_state_size, len(index_to_char))

        encoder_h0 = np.random.uniform(-0.2, 0.2, encoder_hidden_state_size)
        decoder_h0 = np.random.uniform(-0.2, 0.2, len(index_to_char))

        end_of_sequence = np.ones(len(index_to_char)) * -1

        for epoch in range(10000):
            debug = epoch % 100 == 0
            for word_vectors, char_vectors in random.sample(training_set, 30):
                encoder_results = {}
                encoded_state = encoder.forward_prop(word_vectors, encoder_h0, encoder_results)[-1]

                decoder_results = {}

                # FIXME!
                def decoder_input_generator():
                    yield np.zeros(len(index_to_char) + encoder_hidden_state_size)

                    prev_h = decoder_results["hs"][-1]
                    resulting_char = char_from_vector(prev_h)
                    if resulting_char is not " ":
                        yield np.concatenate([vector_from_char(resulting_char), encoded_state])

                # hs = decoder.forward_prop(decoder_input_generator(), decoder_h0, decoder_results)
                hs = decoder.forward_prop(decoder_input_generator(), encoded_state, decoder_results)

                if len(hs) <= len(char_vectors):
                    targets = char_vectors[:len(hs)]
                else:
                    targets = char_vectors + [end_of_sequence for _ in range(len(hs) - len(char_vectors))]

                decoder_errors = [h - target for h, target in zip(hs, targets)]

                encoded_state_error = decoder.back_prop(decoder_errors, decoder_results)
                decoder.train(0.1, decoder_results)

                # FIXME!
                # encoded_state_error = np.zeros(encoder_hidden_state_size)
                encoder_errors = ([np.zeros(encoder_hidden_state_size)] * (len(word_vectors) - 1)) + [encoded_state_error]

                encoder.back_prop(encoder_errors, encoder_results)
                encoder.train(0.1, encoder_results)

                if debug:
                    print(" ".join([word_from_vector(word_vector) for word_vector in word_vectors]))
                    print("".join([char_from_vector(h) for h in hs]))
                    print(sum([np.sum(np.square(err)) for err in decoder_errors]))
                    debug = False







