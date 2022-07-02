# wxy
def getText():
    with open("train.txt", "w", encoding='utf-8') as writ:
        with open("corpus.tc.en", "r", encoding='utf-8') as file:
            for line in file:
                tmp = line.strip()
                if len(tmp.split()) <= 128:
                    writ.write(tmp)
                    writ.write('\n')
if __name__ == "__main__":
    getText()