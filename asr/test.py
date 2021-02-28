"""
Test function of the asr pipeline.
Adapted from https://colab.research.google.com/drive/1IPpwx4rX32rqHKpLz7dc8sOKspUa-YKO
"""

from tqdm import tqdm
import torch
import torch.nn.functional as F
from asr.utils import GreedyDecoder
from asr.data import TextTransform
from asr.metrics import cer, wer


def test(args, model, test_loader, criterion, logger):
    print('\nevaluating...')
    model.eval()
    test_loss = 0
    test_cer, test_wer = [], []
    with torch.no_grad():
        for i, _data in tqdm(enumerate(test_loader)):
            spectrograms, labels, input_lengths, label_lengths = _data
            spectrograms, labels = spectrograms.to(args.device), labels.to(args.device)

            output = model(spectrograms)  # (batch, time, n_class)
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1)  # (time, batch, n_class)

            loss = criterion(output, labels, input_lengths, label_lengths)
            test_loss += loss.item() / len(test_loader)

            text_transform = TextTransform()
            decoded_preds, decoded_targets = GreedyDecoder(
                output.transpose(0, 1), text_transform, labels, label_lengths)
            for j in range(len(decoded_preds)):
                test_cer.append(cer(decoded_targets[j], decoded_preds[j]))
                test_wer.append(wer(decoded_targets[j], decoded_preds[j]))

    avg_cer = sum(test_cer) / len(test_cer)
    avg_wer = sum(test_wer) / len(test_wer)
    metric_dict = {'test_loss': test_loss, 'test_cer': avg_cer, 'test_wer': avg_wer}
    logger.log_metrics(metric_dict)
    print('Test set: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f}\n'.format(test_loss, avg_cer, avg_wer))

    return metric_dict
