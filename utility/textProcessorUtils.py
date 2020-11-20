import re

from data.emoticons import EMOTICONS, UNICODE_EMO


def convert_emoticons(text):
    """Function to convert emoticons
    @param text: String of text
    @return: Cleaned text
    """
    try:
        for emot in EMOTICONS:
            text = re.sub(u'(' + emot + ')', "_".join(EMOTICONS[emot].replace(",", "").split()), text)
        return text
    except:
        return text


def convert_emojis(text):
    """Function to convert the emoticons
    @param text: String of text
    @return: Cleaned text
    """
    try:
        for emot in UNICODE_EMO:
            text = re.sub(r'(' + emot + ')', "_".join(UNICODE_EMO[emot].replace(",", "").replace(":", "").split()),
                          text)
        return text
    except:
        return text


def remove_emoji(text):
    """Function to remove the emojis
    @param text: String of text
    @return: Cleaned text
    """
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def remove_emoticons(text):
    """Function to remove the emoticons
    @param text: String of text
    @return: Cleaned text
    """
    emoticon_pattern = re.compile(u'(' + u'|'.join(k for k in EMOTICONS) + u')')
    return emoticon_pattern.sub(r'', text)