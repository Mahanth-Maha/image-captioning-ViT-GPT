import re
import unicodedata

def preprocess(text):
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9'.,!?;:()\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

def preprocess_mytlm_tokenizer(text, unicode_norm = False, lower_case = False, remove_punc = False, insert_space = True , preprocess_mode = 'custom'):
    
    # irreversible
    if remove_punc:
        text = re.sub(r"[^a-z0-9'.,!?;:()\-\s]", " ", text)
    
    # irreversible
    if unicode_norm or preprocess_mode == 'all':
        text = unicodedata.normalize("NFKC", text)
    
    # irreversible
    if lower_case or preprocess_mode == 'all':
        text = text.lower()
    
    # reversible
    if insert_space or preprocess_mode == 'all':
        text = ' ' + text
        text = re.sub(r'([\n\r\t\v\f]+)', r' \1 ', text)
        text = re.sub(r'([.,!?;:(){}\[\]\"\'\-])', r' \1', text) 
        text = re.sub(r'(?<!\s)(?=\b\w)', ' ', text)
        
    text = re.sub(r'\s+', ' ', text)
    return text

def preprocess_captions(text):
    return preprocess_mytlm_tokenizer(text, unicode_norm = True, lower_case = True, remove_punc = False, insert_space = True)


# reversible : insert space only :: 
def postprocess_captions(text):
    text = text.strip()
    text = re.sub(r'\s+([.,!?;:)\]])', r'\1', text)
    text = re.sub(r'([\(\[\{])\s+', r'\1', text)
    text = re.sub(r"\s+([\"'])", r"\1", text)
    text = re.sub(r"([\"'])\s+", r"\1", text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
