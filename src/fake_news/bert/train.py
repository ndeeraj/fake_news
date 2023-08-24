from datasets import load_metric
import os
import numpy as np
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    BertConfig,
    Trainer, TrainingArguments
)
from fake_news.bert import constants as C
from fake_news.bert.data import FakeNewsDataset, train_test


os.putenv('WANDB_DISABLED', 'YES')

f1 = load_metric('f1', 'f1')
prec = load_metric('precision', 'precision')
rec = load_metric('recall', 'recall')


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        'prec': prec.compute(predictions=predictions, references=labels),
        'rec': rec.compute(predictions=predictions, references=labels),
        'f1': f1.compute(predictions=predictions, references=labels),
    }


if __name__ == '__main__':

    config = BertConfig(num_labels=C.n_classes)
    model = BertForSequenceClassification.from_pretrained(
        C.model_name,
        # config=config
        num_labels=C.n_classes,
    )
    tokenizer = BertTokenizer.from_pretrained(C.model_name)

    train, test = train_test()

    metric_name = 'f1'
    args = TrainingArguments(
        './models_save',
        per_device_train_batch_size=C.micro_batch_sz,
        per_device_eval_batch_size=C.micro_batch_sz,
        gradient_accumulation_steps=C.accum_steps,
        learning_rate=C.lr,
        metric_for_best_model=metric_name,
        lr_scheduler_type='linear',
        warmup_steps=250,
        report_to=['tensorboard'],
        logging_steps=10,
        eval_steps=500,
    )

    train_ds = FakeNewsDataset(train, tokenizer)
    test_ds = FakeNewsDataset(test, tokenizer)

    trainer = Trainer(
        model,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=train_ds.collator,
        compute_metrics=compute_metrics,
        args=args
    )

    hist = trainer.train()
    print(hist)

