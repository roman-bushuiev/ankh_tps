import copy
import logging
import sys
import itertools
import ankh
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_scheduler
from tqdm.auto import tqdm
from Bio import SeqIO


def setup_logger(log_file_path=None, log_name='log'):

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    if logger.hasHandlers():
        logger.handlers.clear()

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    if log_file_path:
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class MaskSeqDataset(Dataset):
    def __init__(self, fasta_pth, tokenizer, seq_len=512, mask_p=0.2):
        self.seqs = list(SeqIO.parse(open(fasta_pth), 'fasta'))
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.mask_p = mask_p
        self.mask_tokens = tokenizer.additional_special_tokens

    def __len__(self):
        return len(self.seqs)

    def _tokenize(self, seq):
        return self.tokenizer.encode_plus(
            seq,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=self.seq_len,
            return_tensors='pt'
        )

    def _mask(self, seq: str, exact: bool = True) -> tuple[str, str]:
        """
        Generate unsupervised training example for a sequence according to Experiment 4
        in the Ankh paper (unigram T5 span masking).
        """
        seq = np.array(list(seq), dtype='object')
        n = len(seq)

        # Generate unigram mask
        if exact:  # mask exactly self.mask_p % tokens
            mask = np.full(n, False)
            mask[:int(self.mask_p * n)] = True
            mask = mask[np.random.permutation(n)]
        else:  # approximate
            mask = np.rand(n) < self.mask_p

        # Get unique sentinel for each masked token
        assert len(self.mask_tokens) >= mask.sum(), 'Too many tokens to mask.'
        sentinels = self.mask_tokens[:mask.sum()]

        # Construct label
        y = list(itertools.chain.from_iterable([pair for pair in zip(sentinels, seq[mask])]))

        # Construct input (mask)
        x = copy.deepcopy(seq)
        x[mask] = sentinels

        x, y = ''.join(x), ''.join(y)
        return x, y

    def __getitem__(self, i):

        seq = str(self.seqs[i].seq)[:self.seq_len]
        if seq[-1] == '*':
            seq = seq[:-1]

        seq_mask, label = self._mask(seq)
        seq_mask = self._tokenize(seq_mask)
        label = self._tokenize(label)

        label_ids = label['input_ids']
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100

        return dict(
            input_ids=seq_mask['input_ids'].flatten(),
            attention_mask=seq_mask['attention_mask'].flatten(),
            labels=label_ids.flatten(),
            decoder_attention_mask=label['attention_mask'].flatten()
        )


def main():

    logger = setup_logger('train_ankh_tps.log')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, tokenizer = ankh.load_base_model(generation=True)
    special_token_ids = torch.tensor([i for t, i in tokenizer.get_vocab().items() if t.startswith('<')], device=device)
    logger.info(f'Model size: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M parameters')

    dataset = MaskSeqDataset('data/tsa_pfam_supfam_common.fasta', tokenizer, seq_len=512)
    logger.info(f'Dataset size: {len(dataset)}')
    dataloader = DataLoader(dataset, shuffle=True, batch_size=4)

    num_epochs = 100
    num_training_steps = num_epochs * len(dataloader)
    progress_bar = tqdm(range(num_training_steps))

    optimizer = AdamW(model.parameters())
    lr_scheduler = get_scheduler(
        'linear',
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    model = model.to(device)

    model.train()
    for epoch in range(num_epochs):
        for i, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            logits = outputs.logits

            y_true = batch['labels']
            y_pred = logits.argmax(dim=-1)
            acc_mask = ~torch.isin(y_true, special_token_ids)
            acc = (y_pred == y_true)[acc_mask].sum() / acc_mask.sum()

            loss = outputs.loss
            if i % 100 == 0:
                logger.info(f'Train loss: {loss.item()}, Train acc: {acc}')
            loss.backward()

            optimizer.step()
            lr_scheduler.step()

            optimizer.zero_grad()
            progress_bar.update()

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, f'./tps_ankh.pth')

        logger.info(f'epoch: {epoch + 1} -- loss: {loss}')


if __name__ == '__main__':
    main()
