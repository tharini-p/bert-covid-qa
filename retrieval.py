import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def segment_documents(docs, max_doc_length=450):
  # List containing full and segmented docs
  segmented_docs = []

  for doc in docs:
    # Split document by spaces to obtain a word count that roughly approximates the token count
    split_to_words = doc.split(" ")

    # If the document is longer than our maximum length, split it up into smaller segments and add them to the list 
    if len(split_to_words) > max_doc_length:
      for doc_segment in range(0, len(split_to_words), max_doc_length):
        segmented_docs.append( " ".join(split_to_words[doc_segment:doc_segment + max_doc_length]))

    # If the document is shorter than our maximum length, add it to the list
    else:
      segmented_docs.append(doc)

  return segmented_docs

def get_top_k_articles(query, docs, k=2):

  # Initialize a vectorizer that removes English stop words
  vectorizer = TfidfVectorizer(analyzer="word", stop_words='english')

  # Create a corpus of query and documents and convert to TFIDF vectors
  query_and_docs = [query] + docs
  matrix = vectorizer.fit_transform(query_and_docs)

  # Holds our cosine similarity scores
  scores = []

  # The first vector is our query text, so compute the similarity of our query against all document vectors
  for i in range(1, len(query_and_docs)):
    scores.append(cosine_similarity(matrix[0], matrix[i])[0][0])

  # Sort list of scores and return the top k highest scoring documents
  sorted_list = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
  top_doc_indices = [x[0] for x in sorted_list[:k]]
  top_docs = [docs[x] for x in top_doc_indices]
  
  return top_docs

def answer_question(model, tokenizer, question, answer_text, device):

    input_ids = tokenizer.encode(question, answer_text, max_length=512)
    
    # ======== Set Segment IDs ========
    # Search the input_ids for the first instance of the `[SEP]` token.
    sep_index = input_ids.index(tokenizer.sep_token_id)

    # The number of segment A tokens includes the [SEP] token istelf.
    num_seg_a = sep_index + 1

    # The remainder are segment B.
    num_seg_b = len(input_ids) - num_seg_a

    # Construct the list of 0s and 1s.
    segment_ids = [0]*num_seg_a + [1]*num_seg_b

    # There should be a segment_id for every input token.
    assert len(segment_ids) == len(input_ids)

    outputs = model(torch.tensor([input_ids]).to(device), # The tokens representing our input text.
                    token_type_ids=torch.tensor([segment_ids]).to(device), # The segment IDs to differentiate question from answer_text
                    return_dict=True) 

    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    # ======== Reconstruct Answer ========
    # Find the tokens with the highest `start` and `end` scores.
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)

    # Get the string versions of the input tokens.
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Start with the first token.
    answer = tokens[answer_start]

    # Select the remaining answer tokens and join them with whitespace.
    for i in range(answer_start + 1, answer_end + 1):
        
        # If it's a subword token, then recombine it with the previous token.
        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]
        
        # Otherwise, add a space then the token.
        else:
            answer += ' ' + tokens[i]

    return answer