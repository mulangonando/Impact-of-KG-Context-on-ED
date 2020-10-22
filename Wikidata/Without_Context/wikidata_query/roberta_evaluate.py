from sklearn.metrics import mean_squared_error, matthews_corrcoef, confusion_matrix
from scipy.stats import pearsonr
import os
import logging
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import random
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm_notebook, trange
from tensorboardX import SummaryWriter
import numpy as np
import pandas as pd

from pytorch_transformers import AdamW, WarmupLinearSchedule

from wikidata_query.utils_transformer import (convert_examples_to_features,
                        output_modes, processors)

from pytorch_transformers import (WEIGHTS_NAME, RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)


_path = os.path.dirname(__file__)
_saving_dir = os.path.join(_path, '../data/')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
_bucket_size = 10
_minimum_trace = 10
# Create a TransformerModel
#model = TransformerModel('roberta', 'roberta-base')

args = {
    'data_dir': '/home/IAIS/cprabhu/Thesis/ned-graphs/wikidata_entity_linking_with_attentive_rnn_triplets_transformer_without_context_entity/wikidata_query/data/',
    'model_type':  'roberta',
    'model_name': 'roberta-base',
    'task_name': 'binary',
    'output_dir': '/home/IAIS/cprabhu/Thesis/ned-graphs/wikidata_entity_linking_with_attentive_rnn_triplets_transformer_without_context_entity/wikidata_query/outputs/',
    'cache_dir': '/home/IAIS/cprabhu/Thesis/ned-graphs/wikidata_entity_linking_with_attentive_rnn_triplets_transformer_without_context_entity/wikidata_query/cache/',
    'do_train': True,
    'do_eval': True,
    'fp16': False,
    'fp16_opt_level': 'O1',
    'max_seq_length': 512,
    'output_mode': 'classification',
    'train_batch_size': 10,
    'eval_batch_size': 10,

    'gradient_accumulation_steps': 1,
    'num_train_epochs': 1,
    'weight_decay': 0,
    'learning_rate': 4e-5,
    'adam_epsilon': 1e-8,
    'warmup_ratio': 0.06,
    'warmup_steps': 0,
    'max_grad_norm': 1.0,

    'logging_steps': 50,
    'evaluate_during_training': False,
    'save_steps': 2000,
    'eval_all_checkpoints': True,

    'overwrite_output_dir': False,
    'reprocess_input_data': True,
    'notes': 'Using Wikidata dataset'
}

MODEL_CLASSES = {
    #'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    #'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    #'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
}
config_class, model_class, tokenizer_class = MODEL_CLASSES[args['model_type']]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = config_class.from_pretrained(args['model_name'], num_labels=2, finetuning_task=args['task_name'])
tokenizer = tokenizer_class.from_pretrained(args['model_name'])
model = model_class.from_pretrained(args['model_name'])
model.to(device)

task = args['task_name']

if task in processors.keys() and task in output_modes.keys():
    processor = processors[task]()
    label_list = processor.get_labels()
    num_labels = len(label_list)
else:
    raise KeyError(f'{task} not found in processors or in output_modes. Please check utils.py.')

def get_mismatched(labels, preds):
    mismatched = labels != preds
    examples = processor.get_dev_examples(args['data_dir'])
    wrong = [i for (i, v) in zip(examples, mismatched) if v]
    
    return wrong

def get_eval_report(labels, preds):
    mcc = matthews_corrcoef(labels, preds)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    try:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * 1 / (1 / precision + 1 / recall)
        logger.info(" Precision ", str(precision))
        logger.info(" Recall ", str(recall))
        logger.info(" F1 ", str(f1))
        print(" tp ", tp)
        print(" tn ", tn)
        print(" fp ", fp)
        print(" fn ", fn)
        print(" mcc ", mcc)
        print(" Precision ", precision)
        print(" Recall ", recall)
        print(" F1 ", f1)
        sys.stdout.flush()
    except:
        print('Cannot compute precision and recall.')
    return {
        "mcc": mcc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    }, get_mismatched(labels, preds)

def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    return get_eval_report(labels, preds)

def evaluate(eval_dataset, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args['output_dir']

    results = {}
    EVAL_TASK = args['task_name']

    #eval_dataset = load_and_cache_examples(EVAL_TASK, tokenizer, evaluate=True)
    #if not os.path.exists(eval_output_dir):
    #    os.makedirs(eval_output_dir)


    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args['eval_batch_size'])

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args['eval_batch_size'])
    print("***** Running evaluation {} *****".format(prefix))
    print("  Num examples = %d", len(eval_dataset))
    print("  Batch size = %d", args['eval_batch_size'])
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm_notebook(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args['model_type'] in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
                      'labels':         batch[3]}
            torch.cuda.empty_cache()
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    if args['output_mode'] == "classification":
        preds = np.argmax(preds, axis=1)
    elif args['output_mode'] == "regression":
        preds = np.squeeze(preds)
    result, wrong = compute_metrics(EVAL_TASK, preds, out_label_ids)
    results.update(result)

    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return results, wrong