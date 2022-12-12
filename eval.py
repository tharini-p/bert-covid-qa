
import regex as re
import string
from collections import Counter
from tqdm import trange

import torch


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def wrap(s, w):
    return [s[i:i + w] for i in range(0, len(s), w)]

def eval_qa(question, text, originalAnswer, maxLength, tokenizer, model, device):
    texts = wrap(text, maxLength)
    fScore = None
    result_answer = None
    for subtext in texts:
      encoding = tokenizer.encode_plus(question, subtext, return_tensors="pt")
      input_ids = encoding["input_ids"].to(device)
      attention_mask = encoding["attention_mask"].to(device)

      start_scores, end_scores, attn_scores = model(input_ids, attention_mask=attention_mask,return_dict=False)
      all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

      answer_tokens = all_tokens[torch.argmax(start_scores) :torch.argmax(end_scores)+1]
      answer = tokenizer.decode(tokenizer.convert_tokens_to_ids(answer_tokens))
      f1=f1_score(answer,originalAnswer)
      if fScore == None or fScore < f1:
        result_answer = answer
        result_attn_scores = attn_scores
        best_context = subtext
        result_input_ids = input_ids
        fScore = f1
      
    return result_answer, result_attn_scores, result_input_ids, best_context


def evaluate_model(model, val_data, tokenizer, device, answer_getter, max_seqlen=512):
  val_contexts, val_questions, val_answers = val_data
  fiScoresAll = []
  emScoresAll = []
  almostScoresAll = []
  almostCountAll = []
  answerPresentAll = []
  answerPresentCountAll = []

  p,ans=[],[]
  f1_total=0
  em_total=0
  almost_total_count = 0
  almost_total = 0
  answerPresent_total_count = 0
  answerPresent_total = 0
  count=0
  for i in trange(len(val_contexts)):
    answer = answer_getter(val_answers[i])
    pred, _, _, _ = eval_qa(val_questions[i], val_contexts[i], answer, 512, tokenizer, model, device)
    p.append(pred)
    ans.append(answer)
    f1=f1_score(pred,answer)
    em=exact_match_score(pred,answer)

    if f1 >= 0.5:
      almost_total += f1
      almost_total_count += 1
    if f1 > 0:
      answerPresent_total += f1
      answerPresent_total_count += 1
    if em:
      em_total+=1
    
    f1_total+=f1
    count+=1
      
  f1_total=f1_total/count
  em_total=em_total/count
  almost_total = almost_total/count
  almost_total_count = almost_total_count/count
  answerPresent_total = answerPresent_total/count
  answerPresent_total_count = answerPresent_total_count/count

  fiScoresAll.append(f1_total)
  emScoresAll.append(em_total)
  almostScoresAll.append(almost_total)
  almostCountAll.append(almost_total_count)
  answerPresentAll.append(answerPresent_total)
  answerPresentCountAll.append(answerPresent_total_count)

  return em_total, f1_total
