import copy
import logging
import sys
import itertools
import ankh
import torch
import wandb
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

        # Preprocess sequence
        seq = str(self.seqs[i].seq)[:self.seq_len]
        if seq[-1] == '*':
            seq = seq[:-1]

        # Mask and tokenize sequence
        seq_mask, label = self._mask(seq)
        seq_mask = self._tokenize(seq_mask)
        label = self._tokenize(label)

        # Prevent cross entropy computation on pad tokens (torch ignores -100 token ids by default)
        label_ids = label['input_ids']
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100

        return dict(
            input_ids=seq_mask['input_ids'].flatten(),
            attention_mask=seq_mask['attention_mask'].flatten(),
            labels=label_ids.flatten(),
            decoder_attention_mask=label['attention_mask'].flatten()
        )


def main():

    num_devices = 8
    lr = 1e-5
    batch_size = 4 * num_devices
    seq_len = 512
    val_frac = 0.1
    dataset_pth = 'data/TPS_mining_v2/all_pfam_supfam_pool_unique.fasta'
    run_name = f'lr={lr}_bs={batch_size}'
    log_pth = f'train_ankh_{run_name}.log'
    num_epochs = 30

    logger = setup_logger(log_pth)
    wandb.init(project='ankh_tps', name=run_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load pre-trained Ankh base
    model, tokenizer = ankh.load_base_model(generation=True)
    special_token_ids = torch.tensor(
        [i for t, i in tokenizer.get_vocab().items() if t.startswith('<')] + [-100]
    , device=device)
    logger.info(f'Model size: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M parameters')
    model = model.to(device)
    if num_devices > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(num_devices)))

    # Define dataset
    dataset = MaskSeqDataset(dataset_pth, tokenizer, seq_len=seq_len)
    val_size = round(val_frac * len(dataset))
    train_set, val_set = torch.utils.data.random_split(dataset, [len(dataset) - val_size, val_size])
    logger.info(f'Train dataset size: {len(train_set)}, Validation dataset size {len(val_set)}')
    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    # Setup optimizer
    num_training_steps = num_epochs * len(train_loader)
    progress_bar = tqdm(range(num_training_steps))
    optimizer = AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_scheduler(
        'linear',
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    def forward(batch, i, val):
        """ Forward pass of the model """

        batch = {k: v.to(device) for k, v in batch.items()}

        # Compute loss
        outputs = model(**batch)
        logits = outputs.logits
        loss = outputs.loss.mean()  # mean is needed for DataParallel?

        # Compute accuracy
        y_true = batch['labels']
        y_pred = logits.argmax(dim=-1)
        acc_mask = ~torch.isin(y_true, special_token_ids)
        acc = (y_pred == y_true)[acc_mask].sum() / acc_mask.sum()

        # Log
        log_prefix = 'Val' if val else 'Train'
        if i % 10 == 0:
            # logger.info(f'[{i} {log_prefix}] loss: {loss.item()}, acc: {acc}')
            wandb.log({f'{log_prefix} loss': loss.item(), f'{log_prefix} accuracy': acc})
        return loss

    for epoch in range(num_epochs):

        # Train
        model.train()
        for i, batch in enumerate(train_loader):

            loss = forward(batch, i, val=False)
            loss.backward()

            optimizer.step()
            lr_scheduler.step()

            optimizer.zero_grad()
            progress_bar.update()

        # Validation
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                loss = forward(batch, i, val=True)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, f'./tps_ankh.pth')

        logger.info(f'epoch: {epoch + 1} -- loss: {loss}')


if __name__ == '__main__':
    main()
