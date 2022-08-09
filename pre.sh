#!/bin/bash
echo "pre-processing test data..."      # 预处理测试语料
src=en
tgt=de
#
for l in $src $tgt; do
    if [ "$l" == "$src" ]; then      
        t="src"
    else
        t="ref"
    fi
    grep '<seg id' test-full/newstest2014-deen-$t.$l.sgm | \      #这一块操作没看懂
        sed -e 's/<seg id="[0-9]*">\s*//g' | \      
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" | \
    perl moses/mosesdecoder\scripts\tokenizer\normalize-punctuation.perl -l $l | \
    perl moses/mosesdecoder\scripts\tokenizer\tokenizer.perl -a -l $l > test.$l      # 分词
    echo "Tokenise成功"
done

#### Train truecaser and truecase
for lang in $src $tgt; do
  perl moses/mosesdecoder/scripts/recaser/train-truecaser.perl -corpus test.$lang -model truecase-test.$lang
  perl moses/mosesdecoder/scripts/recaser/truecase.perl -model truecase-test.$lang < test.$lang > test.tc.$lang
  echo "truecase成功"
done

### bpe
for lang in $src $tgt; do
  subword-nmt apply-bpe -c codes_file.$lang --vocabulary  vocab_file.$lang --vocabulary-threshold 10 --input test.tc.$lang --output test.BPE.$lang
  echo "BPE成功" 
done
