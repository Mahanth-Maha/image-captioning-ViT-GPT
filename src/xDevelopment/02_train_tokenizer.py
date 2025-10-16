import os
import re
import argparse
from tqdm import tqdm
from collections import Counter


import os
import sys
from dotenv import load_dotenv
load_dotenv()

root_dir = os.getenv('PROJECT_ROOT')
root_coco = os.getenv('DATA_DIR')
train_coco = os.getenv('TRAIN_DIR')
val_coco = os.getenv('VAL_DIR')
tokenizer_models_dir = os.getenv('TOK_DIR')

src_dir = os.path.join(root_dir, "src")
train_folder = os.path.join(root_coco, f"{train_coco}")
val_folder = os.path.join(root_coco, f"{val_coco}")
train_ann_file = os.path.join(root_coco, "annotations", f"captions_{train_coco}.json")
val_ann_file = os.path.join(root_coco, "annotations", f"captions_{val_coco}.json")


if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
os.chdir(src_dir)



from datasets.tokenizer import myTokenizer_faster

TOKEN_UNK = '<|unknown|>'
TOKEN_SOS = '<|startofseq|>'
TOKEN_EOS = '<|endofseq|>'
TOKEN_PAD = '<|padding|>'
TOKEN_END = '<|endoftext|>'


print_dash = 75

configs = {
    '512':( 2**9,  2**5, 2*9),
    '1k' :(2**10,  2**6, 2*9),
    '2k' :(2**11,  2**7, 2*8),
    '4k' :(2**12,  2**8, 2*8),
    '8k' :(2**13,  2**9, 2*8),
    '16k':(2**14, 2**10, 2*7),
    '32k':(2**15, 2**11, 2*7),
}


# vocab_size = 2**13
# max_protected_words = 2**10
# max_freq_protect = 2*9

# 1K tokenizer
# vocab_size = 2**9
# max_protected_words = 2**5
# max_freq_protect = 2*9


def parse_args():
    parser = argparse.ArgumentParser(description="Tokenizer")
    parser.add_argument('-t', "--train", action="store_true",
                        help="starts training")
    parser.add_argument('-n', "--no-test", action="store_false", 
                        default=True, help="no testing")
    parser.add_argument('-v', '--vocab-size', type=int,
                        default=None, help="Vocab Size")
    parser.add_argument('-p', '--max-protected-words', type=int, 
                        default=None, help="max protected words")
    parser.add_argument('-f', '--max-freq-protect', type=int,
                        default=None, help="max freq protect")
    parser.add_argument('-c', '--preset-config', type=str,
                        default=None, help="configs: 512, 1k, 2k, 4k, 8k, 16k, 32k")
    parser.add_argument('-m', '--model-name', type=str,
                        default='bpe', help="configs: bpe, caps_bpe, books_bpe")

    return parser.parse_args()



def main():
    args = parse_args()
    vocab_size, max_protected_words, max_freq_protect = configs.get(args.preset_config, (args.vocab_size, args.max_protected_words, args.max_freq_protect))

    print('info:')
    print(f'Vocab Size : {vocab_size}')

    protect_words = []
    if args.train:
        print('➡️ Tokenizing...')
        all_txt = ''
        if 'caps' in args.model_name or args.model_name == 'bpe':
            with open(f"{tokenizer_models_dir}/train_caps.txt", encoding='utf-8') as f:
                coco_txt = f.read()
            all_txt += coco_txt
            
        if 'books' in args.model_name or args.model_name == 'bpe':
            with open(f"{tokenizer_models_dir}/books_dataset_preprocessed.txt", encoding="utf-8") as f:
                book_txts = f.read()
            all_txt += book_txts

        if all_txt == '':
            print(f'No Data to train')
            return
        
        print(f'Data Loaded')
        all_txt = ' ' + all_txt
        all_txt = re.sub(r'([\n\r\t\v\f]+)', r' \1 ', all_txt)
        all_txt = re.sub(r'([.,!?;:(){}\[\]\"\'\-])', r' \1', all_txt)
        all_txt = re.sub(r'(?<!\s)(?=\b\w)', ' ', all_txt)
        all_txt = re.sub(r'\s+', ' ', all_txt)

        words = all_txt.split()

        print(f'No of words in text {len(words)}')
        print(f'No of unique words in text {len(set(words))}')
        counter = Counter(words)
        for k, v in counter.items():
            if v >= max_freq_protect:
                protect_words.append(k)
            if len(protect_words) >= max_protected_words:
                break
        print(f'No of Protected words: {len(protect_words)}')

    tokenizer = myTokenizer_faster(
        token_model_file=f"{tokenizer_models_dir}/{args.model_name}{vocab_size}.model",
        special_tokens={
            TOKEN_UNK: vocab_size - 5,
            TOKEN_SOS: vocab_size - 4,
            TOKEN_EOS: vocab_size - 3,
            TOKEN_PAD: vocab_size - 2,
            TOKEN_END: vocab_size - 1
        },
        protected_words=protect_words,
        preprocess_mode='removeall'
    )

    if args.train:
        tokenizer.train(all_txt, vocab_size)
    if args.no_test:
        tokenizer.load()
        print()
        print('-'*print_dash)
        print(f'\t\tTesting 1:')
        print('-'*print_dash)
        x = 'This is not a drill.'
        print(f'{x=}')
        print(f'{tokenizer.encode(x)=}')
        decc = tokenizer.decode(tokenizer.encode(x))
        print(f'{tokenizer.decode(tokenizer.encode(x))=}\n')
        print(f'Are they same ?: (Short ans:{"✅ Yes" if x == decc else "❌ No"})\n')
        for i, j in zip(x, decc):
            if i != j:
                print(f'failed here: text = {i} out put = {j}')
                break
        else:
            print(f'They are Same')

        print()
        print('-'*print_dash)
        print(f'\t\tTesting 2:')
        print('-'*print_dash)
        x = ' with light flowing in the window. little girl in blue dress standing in a bathroom. a fighter'
        print(f'{x=}')
        encc = tokenizer.encode(x)
        print(f'{tokenizer.encode(x)=}')
        decc = tokenizer.decode(encc)
        print(f'{tokenizer.decode(tokenizer.encode(x))=}')
        print(f'Are they same ?: (Short ans:{"✅ Yes" if x == decc else "❌ No"})\n')
        for i, j in zip(x, decc):
            if i != j:
                print(f'failed here: text = {i} out put = {j}')
                break
        else:
            print(f'They are Same')
        print()
        print('-'*print_dash)
        print()
    print('✍️ Done!')

if __name__=='__main__':
    main()

# mkdir -p logs/
# nohup python 02_train_tokenizer.py -c 512 > logs/training_tokenizer_bpe512_load.log  2>&1 &
# nohup python 02_train_tokenizer.py -c 1k > logs/training_tokenizer_bpe1k_load.log  2>&1 &
# nohup python 02_train_tokenizer.py -c 2k > logs/training_tokenizer_bpe2k_load.log  2>&1 &
# nohup python 02_train_tokenizer.py -c 4k > logs/training_tokenizer_bpe4k_load.log  2>&1 &
# nohup python 02_train_tokenizer.py -c 8k > logs/training_tokenizer_bpe8k_load.log  2>&1 &
# nohup python 02_train_tokenizer.py -c 16k > logs/training_tokenizer_bpe16k_load.log  2>&1 &
# nohup python 02_train_tokenizer.py -c 32k > logs/training_tokenizer_bpe32k_load.log  2>&1 &

# nohup python 02_train_tokenizer.py -t -c 512 > logs/training_tokenizer_bpe512_train.log  2>&1 &
# nohup python 02_train_tokenizer.py -t -c 1k > logs/training_tokenizer_bpe1k_train.log  2>&1 &
# nohup python 02_train_tokenizer.py -t -c 2k > logs/training_tokenizer_bpe2k_train.log  2>&1 &
# nohup python 02_train_tokenizer.py -t -c 4k > logs/training_tokenizer_bpe4k_train.log  2>&1 &
# nohup python 02_train_tokenizer.py -t -c 8k > logs/training_tokenizer_bpe8k_train.log  2>&1 &
# nohup python 02_train_tokenizer.py -t -c 16k > logs/training_tokenizer_bpe16k_train.log  2>&1 &
# nohup python 02_train_tokenizer.py -t -c 32k > logs/training_tokenizer_bpe32k_train.log  2>&1 &


# nohup python xDevelopment/02_train_tokenizer.py -t -c 16k -m caps_bpe > xDevelopment/logs/training_tokenizer_caps_bpe16k_load.log 2>&1 &
# nohup python xDevelopment/02_train_tokenizer.py -t -c 16k -m books_bpe > xDevelopment/logs/training_tokenizer_books_bpe16k_load.log 2>&1 &
# nohup python -m xDevelopment.02_train_tokenizer -t -c 16k -m caps_bpe > logs/training_tokenizer_caps_bpe16k_load.log 2>&1 &