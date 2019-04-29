import numpy as np
import json


class TextVectorizer:

    def __init__(self, unique_characters, categories):
        """
        TextVectorizer is used to convert plain text into numpy arrays
        that are compatible with RNNs

        Arguments:
            unique_characters {[list/tuple/set]}
            categories {[list/tuple/set]}
        """
        self._char2idx = {}
        self._idx2char = {}
        for idx, char in enumerate(sorted(list(unique_characters))):
            self._idx2char[idx] = char
            self._char2idx[char] = idx

        self._category2idx = {}
        self._idx2category = {}
        for idx, category in enumerate(sorted(list(categories))):
            self._idx2category[idx] = category
            self._category2idx[category] = idx

        self._NUM_CHARS = len(unique_characters)
        self._CHARS = sorted(list(unique_characters))
        self._NUM_CATEGORIES = len(categories)
        self._CATEGORIES = sorted(list(categories))

    def text_to_indices(self, text):
        """
        Converts plain text into indices based on the internal
        dictionary

        Example:
            "abc" --> [64, 65, 66]

        Arguments:
            text {str}

        Returns:
            list
        """
        indices = []
        for char in text:
            indices.append(self._char2idx[char])
        return indices

    def text_to_array(self, text, category=None, dtype=np.float32):
        """
        Converts plain text to a new numpy array.
        Depending whether category argument is provided the shape will
        be either (len(text), len(unique_characters)) or
        (len(text), len(unique_characters) + len(categories))

        Arguments:
            text {str}

        Keyword Arguments:
            category {str}
            dtype {np.dtype} -- the output array will be cast to this type
                (default: {np.float32})

        Raises:
            ValueError: If the given category is not recognized

        Returns:
            np.array
        """
        if category is None:
            shape = (len(text), self._NUM_CHARS)
        elif category not in self._CATEGORIES:
            raise ValueError(f"Incorrect category selected ({category})")
        else:
            shape = (len(text), self._NUM_CHARS + self._NUM_CATEGORIES)

        category_idx = self._category2idx[category] + self._NUM_CHARS
        arr = np.zeros(shape=shape, dtype=dtype)
        for idx, char in enumerate(text):
            arr[idx][self._char2idx[char]] = 1
            if category:
                arr[idx][category_idx] = 1
        return arr

    def indices_to_text(self, indices):
        """
        Converts a list of indices to plain text
        This function is the opposite of .text_to_indices(...)

        Example:
            [64, 65, 66] --> "abc"

        Arguments:
            indices {list}

        Returns:
            str
        """
        chars = []
        for idx in indices:
            chars.append(self._idx2char[idx])
        return "".join(chars)

    def array_to_text(self, array):
        """
        Converts a vectorized representation into plain text
        This function is the opposite of .text_to_array(...)

        Arguments:
            array {np.array}

        Returns:
            str
        """
        chars = []
        for vector in array:
            idx = vector[:self._NUM_CHARS].argmax()
            chars.append(self._idx2char[idx])
        return "".join(chars)

    def repeat_last(self, array, target_len):
        """
        Transforms the given array into shape
        (target_len, array.shape[1])
        by repeating the last element

        This is usefull during embedding

        Arguments:
            array {np.array}
            target_len {int}

        Returns:
            np.array
        """
        if len(array) >= target_len:
            return np.copy(array)

        needed = target_len - len(array)

        last_vector = array[-1]  # 1-D array of shape (array.shape[1],)
        repeated_last_vector = np.tile(last_vector, needed)  # 1-d array of shape (needed * array.shape[1],)
        repeated_last_vector = repeated_last_vector.reshape((needed, -1))  # 2-d array of shape (needed, array.shape[1])

        appended_arr = np.append(np.copy(array), repeated_last_vector, axis=0)
        return appended_arr

    def category_to_index(self, category):
        """
        Converts the given category into an index

        Example:
            "the-nine" --> 113

        Arguments:
            category {str}

        Returns:
            int
        """
        if category not in self._CATEGORIES:
            raise ValueError(f"Incorrect category selected ({category})")
        return self._category2idx[category]

    def index_to_category(self, idx):
        """
        Converts the given index into plain text category
        This function is the opposite of .category_to_index(...)

        Example:
            113 --> "the-nine"

        Arguments:
            idx {int}

        Returns:
            str
        """
        if idx > self._NUM_CATEGORIES:
            raise ValueError(f"Incorrect index ({idx})")
        return self._idx2category[idx]


# ----------------------------------------------------------------------------


class TextGenerator:

    def __init__(self, model, vectorizer):
        """
        Handles text generation

        Arguments:
            model {model.MyModel}
            vectorizer {TextVectorizer}
        """
        self.model = model
        self.vectorizer = vectorizer

    def _randomly_make_selection(self, outputs, bias=[0.0, 0.005, 0.01]):
        """
        Randomly selects an output based on the given
        probabilities. This function can be considered a slightly
        more random version of np.array.argmax()

        Additionally a bias argument can be provided to increase the
        chances of certain less-likely outputs.
        NOTE:
        It's recommended to keep to bias values close to 0.0
        Higher values lead to more unpredictability
        A good default can be [0.0, 0.005, 0.01]

        The amount of outputs that could be sampled is equal to len(bias)
        so it defaults to 3

        Arguments:
            outputs {np.array} -- 1-D or 2-D
            bias {list}

        Raises:
            ValueError: if len(bias) > len(outputs)

        Returns:
            int
        """
        num_outputs = len(bias)

        if len(outputs.shape) == 3:
            outputs = outputs[0][-1]
        elif len(outputs.shape) == 2:
            outputs = outputs[-1]
        else:
            raise ValueError("Incorrect output shape, expected 3-D or 2-D,"
                             f" got {len(outputs.shape)}")

        top_indices = outputs.argsort()[-num_outputs:][::-1]
        top_probs = outputs[top_indices]
        top_probs = np.add(top_probs, bias)
        top_probs /= sum(top_probs)  # now sum(top_probs) == 1
        chosen = np.random.choice(top_indices, p=top_probs)
        return chosen

    def generate(self,
                 start_str,
                 category,
                 char_limit=10000,
                 bias=[0.0, 0.005, 0.01]):
        """
        This generator continuously yields predicted characters and stops
        when the char_limit is reached or the generated character is "␃"

        Arguments:
            start_str {str} -- initial input to the model
            category {str}

        Keyword Arguments:
            char_limit {int} -- upon reaching this limit the generation
                will be stopped despite the last generated character
                (default: {10000})
            bias {list} -- see the docstrig of ._randomly_make_selection(...)
                The default should be fine for most cases
                (default: [0.0, 0.005, 0.01])
        """
        if len(start_str) == 0:
            raise ValueError("A start_str must be provided")

        self.model.forget()

        char = start_str
        for i in range(char_limit):
            arr = self.vectorizer.text_to_array(char, category=category)  # 2-D array
            out = self.model(np.expand_dims(arr, 0),  # add a bias dimension
                             remember=True,
                             dropout_rate=0.0).numpy()  # 3-D array
            if char[-1] in [" ", "\n"]:
                choice = self._randomly_make_selection(out, bias=bias)
            else:
                choice = out[0][-1].argmax()
            char = self.vectorizer.indices_to_text([choice])[0]  # str
            yield(char)
            if char == "␃":
                break


def create_default_vectorizer(char_path='data/default_characters.json',
                              cat_path='data/default_categories.json'):
    """
    Loads the default unique characters and categories
    and returns a new TextVectorizer

    Keyword Arguments:
        char_path {str} --  (default: {'data/default_characters.json'})
        cat_path {str} -- (default: {'data/default_categories.json'})

    Returns:
        TextVectorizer
    """
    with open(char_path, 'r') as f:
        chars = json.load(f)
    with open(cat_path, 'r') as f:
        cats = json.load(f)
    return TextVectorizer(unique_characters=chars,
                          categories=cats.keys())
