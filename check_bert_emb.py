from transformers import BertTokenizer, BertModel
from tokenizers import Tokenizer
import torch

bert_model = BertModel.from_pretrained('bert-base-multilingual-cased')
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
tokenizer_5500 = Tokenizer.from_file(f"data/vibert_5500.json")
tokenizer_6000 = Tokenizer.from_file(f"data/vibert_6000.json")
tokenizer_6500 = Tokenizer.from_file(f"data/vibert_6500.json")
tokenizer_7000 = Tokenizer.from_file(f"data/vibert_7000.json")
tokenizer_7500 = Tokenizer.from_file(f"data/vibert_7500.json")

def add_cls_sep(text):
    return "[CLS] " + text + " [SEP]"

text = "tôi sợ giận quá hóa liều , nó không thèm rủ tôi đi bơi nữa thì khốn ."

# ===============================================================
text_processed = add_cls_sep(text)
tokenized_text = tokenizer.tokenize(text_processed)
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
segments_ids = [0 for i in range(len(indexed_tokens))]
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

with torch.no_grad():
    encoded_layers = bert_model(tokens_tensor, segments_tensors)[0]

bert_embeddings = encoded_layers[0]
bert_embeddings = bert_embeddings[1:(bert_embeddings.size(0)-1)]

# ================================================================
tokenized_text = tokenizer_5500.encode(text)
indexed_tokens = tokenized_text.ids
segments_ids = [0 for i in range(len(indexed_tokens))]
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])
print("5500:", tokenizer.convert_ids_to_tokens(indexed_tokens))
with torch.no_grad():
    encoded_layers = bert_model(tokens_tensor, segments_tensors)[0]

bert_embeddings_5500 = encoded_layers[0]
bert_embeddings_5500 = bert_embeddings_5500[1:(bert_embeddings_5500.size(0)-1)]

# ================================================================
tokenized_text = tokenizer_6000.encode(text)
indexed_tokens = tokenized_text.ids
segments_ids = [0 for i in range(len(indexed_tokens))]
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

with torch.no_grad():
    encoded_layers = bert_model(tokens_tensor, segments_tensors)[0]

bert_embeddings_6000 = encoded_layers[0]
bert_embeddings_6000 = bert_embeddings_6000[1:(bert_embeddings_6000.size(0)-1)]

# ================================================================
tokenized_text = tokenizer_6500.encode(text)
indexed_tokens = tokenized_text.ids
segments_ids = [0 for i in range(len(indexed_tokens))]
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

with torch.no_grad():
    encoded_layers = bert_model(tokens_tensor, segments_tensors)[0]

bert_embeddings_6500 = encoded_layers[0]
bert_embeddings_6500 = bert_embeddings_6500[1:(bert_embeddings_6500.size(0)-1)]

# ================================================================
tokenized_text = tokenizer_7000.encode(text)
indexed_tokens = tokenized_text.ids
segments_ids = [0 for i in range(len(indexed_tokens))]
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

with torch.no_grad():
    encoded_layers = bert_model(tokens_tensor, segments_tensors)[0]

bert_embeddings_7000 = encoded_layers[0]
bert_embeddings_7000 = bert_embeddings_7000[1:(bert_embeddings_7000.size(0)-1)]

# ================================================================
tokenized_text = tokenizer_7500.encode(text)
indexed_tokens = tokenized_text.ids
segments_ids = [0 for i in range(len(indexed_tokens))]
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

with torch.no_grad():
    encoded_layers = bert_model(tokens_tensor, segments_tensors)[0]

bert_embeddings_7500 = encoded_layers[0]
bert_embeddings_7500 = bert_embeddings_7500[1:(bert_embeddings_7500.size(0)-1)]

# print("bert_embeddings:",bert_embeddings)
# print("bert_embeddings_5500:",bert_embeddings_5500)
# print("bert_embeddings_6000:",bert_embeddings_6000)
# print("bert_embeddings_6500:",bert_embeddings_6500)
# print("bert_embeddings_7000:",bert_embeddings_7000)
# print("bert_embeddings_7500:",bert_embeddings_7500)
