#!/bin/bash

src=de
tgt=en
pair=$src-$tgt
#
for lang in $src $tgt; do
    if [ "$lang" == "$src" ]; then      
        t="src"
    else
        t="ref"
    fi
    echo "ti qu"
    grep '<seg id' test-full/newstest2014-deen-$t.$lang.sgm > t_$t.$lang.1.txt
        sed -e 's/<seg id="[0-9]*">\s*//g' t_$t.$lang.1.txt > t_$t.$lang.2.txt      
        sed -e 's/\s*<\/seg>\s*//g' t_$t.$lang.2.txt > t_$t.$lang.3.txt
        sed -e "s/\’/\'/g" t_$t.$lang.3.txt > t_$lang.txt
    echo "ti qu成功"
done
#
echo "pre-processing corpus data..." 
## Tokenise
for lang in $src $tgt; do
  cat \
    commoncrawl.$pair.$lang \
    europarl-v7.$pair.$lang \
    news-commentary-v9.$pair.$lang  |
   perl ../mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l $lang | \
   perl ../moses/mosesdecoder/scripts/tokenizer/tokenizer.perl -a -l $lang  \
   > corpus.tok.$lang
echo "Tokenizer corpus data" 
done
#
#### Clean
perl ../mosesdecoder/scripts/training/clean-corpus-n.perl corpus.tok $src $tgt corpus.clean 3 256 corpus.retained
###
echo "Clean corpus data"
# Tokenise test
for lang in $src $tgt; do   
    perl ../mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l $lang < t_$lang.txt | \
    perl ../mosesdecoder/scripts/tokenizer/tokenizer.perl -a -l $lang  > test.$lang       # 分词
    echo "Tokenise test data"
done


#### Train truecaser and truecase
for lang in $src $tgt; do
  perl ../mosesdecoder/scripts/recaser/train-truecaser.perl -corpus corpus.clean.$lang -model truecase-model.$lang
  perl ../mosesdecoder/scripts/recaser/truecase.perl -model truecase-model.$lang < corpus.clean.$lang > corpus.tc.$lang
  echo "truecaser corpus data"
done

#### Train truecaser and truecase
for lang in $src $tgt; do
  perl ../mosesdecoder/scripts/recaser/train-truecaser.perl -corpus test.$lang -model truecase-test.$lang
  perl ../mosesdecoder/scripts/recaser/truecase.perl -model truecase-test.$lang < test.$lang > test.tc.$lang
  echo "truecase test data"
done

## Tidy up and compress
for lang in $src $tgt; do
  rm -f corpus.clean.$lang corpus.retained corpus.tok.$lang t_src.$lang.1.txt t_src.$lang.2.txt t_src.$lang.3.txt t_src.$lang.4.txt t_ref.$lang.1.txt t_ref.$lang.2.txt t_ref.$lang.3.txt t_ref.$lang.4.txt test.$lang.txt
done

### bpe
for lang in $src $tgt; do
  subword-nmt learn-bpe -s 5000 --input corpus.tc.$lang --output codes_file.$lang
  subword-nmt apply-bpe -c codes_file.$lang --input corpus.tc.$lang | subword-nmt get-vocab --output vocab_file.$lang
  subword-nmt apply-bpe -c codes_file.$lang --vocabulary  vocab_file.$lang --vocabulary-threshold 10 --input corpus.tc.$lang --output BPE.$lang
done

### bpe
for lang in $src $tgt; do
  subword-nmt apply-bpe -c codes_file.$lang --vocabulary  vocab_file.$lang --vocabulary-threshold 10 --input test.tc.$lang --output test.BPE.$lang
  echo "BPE成功" 
done

