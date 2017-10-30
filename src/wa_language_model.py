import re
import random
import pandas as pd
from nltk import bigrams, ConditionalFreqDist


class WhatsappLanguageModel(object):
    """
    Parser for Whatsapp conversation exports

    In order to parse a Whatsapp conversation,
    navigate to the desired conversation and
    export it (WITHOUT! media)

    NOTE:
    The current version only supports german
    conversation logs fully

    Attributes
    ----------
    log: pandas.DataFrame,
        parsed dataframe of
        conversation
    """

    def __init__(self, path):
        """
        Create the Language Model

        Parameter
        ---------
        path: string,
            path locating the conversation file
        """

        # read the file
        raw = open(path, encoding="utf-8").readlines()

        # chunk into column
        # date / time / author / message
        pat_chunk = r"(\d\d\.\d\d\.\d\d), (\d\d\:\d\d\:\d\d): ([^:]*): (.*)"
        chunked = [re.findall(pat_chunk, sent)[0]
                   for sent in raw if re.findall(pat_chunk, sent)]

        # create conversation log
        self.log = pd.DataFrame.from_records(
            chunked,
            columns=["date", "time", "author", "message"]
        )

        # derive the content of the message
        pat_type = r"<(.*) weggelassen>"
        self.log["type"] = self.log["message"].apply(
            lambda x: "Multimedia" if re.findall(
                pat_type, x) else "Text")

        # get list of authors
        self.authors = self.log["author"].unique()

        # fit a bigram model per author
        # fill the dict author -> sentences
        corpora = {}
        for a in self.authors:
            sents = self.log[(self.log["author"] == a) & (
                self.log["type"] == "Text")]["message"].values
            corpora[a] = sents

        # create a conditional fd author -> cfd
        self._author_cfd = {}
        for a in self.authors:
            words = [w.lower() for w in " ".join(corpora[a]).split()]
            self._author_cfd[a] = ConditionalFreqDist(bigrams(words))

    def generate_sentence(self, author, n_words=7):
        """
        Generate a sentence based on learned CFDs

        Parameter
        ---------
        author: string,
            author to me mimed,
            must be in self.authors
        n_words: int, default 7,
            length of the sentence
            to be generated

        Returns
        -------
        sentence: str,
            generated sentence
        """
        sent = []
        word = random.choice(list(self._author_cfd[author].keys()))
        for _ in range(n_words):
            sent.append(word)
            word = self._author_cfd[author][word].max()
        return " ".join(sent)


WALM = WhatsappLanguageModel("./_chat.txt")
