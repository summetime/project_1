#!/bin/bash
echo "pre-processing test data..."      # 预处理测试语料
src=en
tgt=de
#
for lang in $src $tgt; do
    if [ "$lang" == "$src" ]; then      
        t="src"
    else
        t="ref"
    fi
    grep '<seg id' test-full/newstest2014-deen-$t.$lang.sgm > t_$t.$lang.1.txt
        sed -e 's/<seg id="[0-9]*">\s*//g' t_$t.$lang.1.txt > t_$t.$lang.2.txt      
        sed -e 's/\s*<\/seg>\s*//g' t_$t.$lang.2.txt > t_$t.$lang.3.txt
        sed -e "s/\’/\'/g" t_$t.$lang.3.txt > t_$t.$lang.4.txt
    perl moses/mosesdecoder\scripts\tokenizer\normalize-punctuation.perl -l $lang  | \
    perl moses/mosesdecoder\scripts\tokenizer\tokenizer.perl -a -l $lang  > test.$lang       # 分词
    echo "Tokenise成功"
done

#### Train truecaser and truecase
for lang in $src $tgt; do
  perl moses/mosesdecoder/scripts/recaser/train-truecaser.perl -corpus test.$lang -model truecase-test.$lang
  perl moses/mosesdecoder/scripts/recaser/truecase.perl -model truecase-test.$lang < test.$lang > test.tc.$lang
  echo "truecase成功"
done

## Tidy up and compress
for lang in $src $tgt; do
  rm -f t_src.$lang.1.txt t_src.$lang.2.txt t_src.$lang.3.txt t_src.$lang.4.txt t_ref.$lang.1.txt t_ref.$lang.2.txt t_ref.$lang.3.txt t_ref.$lang.4.txt
done

### bpe
for lang in $src $tgt; do
  subword-nmt apply-bpe -c codes_file.$lang --vocabulary  vocab_file.$lang --vocabulary-threshold 10 --input test.tc.$lang --output test.BPE.$lang
  echo "BPE成功" 
done
