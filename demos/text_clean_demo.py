from nlp.preprocess.clean_text import clean_en_text

if __name__ == '__main__':
    sentence = 'This is a good time\' , please be happy'

    print(clean_en_text(sentence))

    path = 'data/imdb/aclImdb.txt'
    path2 = 'data/imdb/aclImdb_a.txt'
    out_lines = []
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

        for line in lines:
            line_arr = line.split("##")
            out_lines.append(line_arr[1].strip() + '##' + line_arr[0].strip() + '\n')
    with open(path2, 'w', encoding='utf-8') as f:
        f.writelines(out_lines)
