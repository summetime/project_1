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
    news-commentary-v12.$pair.$lang  |
   $moses_scripts/tokenizer/normalize-punctuation.perl -l $lang | \
   $moses_scripts/tokenizer/tokenizer.perl -a -l $lang  \
   > corpus.tok.$lang
done
#
###
#### Clean
$moses_scripts/training/clean-corpus-n.perl corpus.tok $src $tgt corpus.clean 3 256 corpus.retained
###
#
#### Train truecaser and truecase
for lang in $src $tgt; do
  $moses_scripts/recaser/train-truecaser.perl -model truecase-model.$lang -corpus corpus.tok.$lang
  $moses_scripts/recaser/truecase.perl < corpus.clean.$lang > corpus.tc.$lang -model truecase-model.$lang
done

## Tidy up and compress
for lang in $src $tgt; do
  gzip corpus.tc.$lang
  rm -f corpus.tok.$lang corpus.clean.$lang corpus.retained
done
