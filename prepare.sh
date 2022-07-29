#!/bin/bash

src=de
tgt=en
pair=$src-$tgt
#
## Tokenise
for lang in $src $tgt; do
  cat \
    commoncrawl.$pair.$lang \
    europarl-v7.$pair.$lang \
    news-commentary-v9.$pair.$lang  |
   perl moses/mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l $lang | \
   perl moses/mosesdecoder/scripts/tokenizer/tokenizer.perl -a -l $lang  \
   > corpus.tok.$lang
done
#
###
#### Clean
perl moses/mosesdecoder/scripts/training/clean-corpus-n.perl corpus.tok $src $tgt corpus.clean 3 256 corpus.retained
###
#
#### Train truecaser and truecase
for lang in $src $tgt; do
  perl moses/mosesdecoder/scripts/recaser/train-truecaser.perl -corpus corpus.tok.$lang -model truecase-model.$lang
  perl moses/mosesdecoder/scripts/recaser/truecase.perl -model truecase-model.$lang < corpus.tok.$lang > corpus.tc.$lang
done

## Tidy up and compress
for lang in $src $tgt; do
  gzip corpus.tc.$lang
  rm -f corpus.tok.$lang corpus.clean.$lang corpus.retained
done

### bpe
for lang in $src $tgt; do
  subword-nmt learn-bpe -s 5000 --input corpus.tc.$lang --output codes_file.$lang
  subword-nmt apply-bpe -c codes_file.$lang --input corpus.tc.$lang | subword-nmt get-vocab --output vocab_file.$lang
  subword-nmt apply-bpe -c codes_file.$lang --vocabulary  vocab_file.$lang --vocabulary-threshold 10 --input corpus.tc.$lang --output BPE.$lang
done

