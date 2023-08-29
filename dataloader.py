import argparse
import csv

def process_spans(spans, text):
    span_text = []
    for span in spans.split(','):
        try:
            start, end = span.split('-')
            span_text.append(text[int(start):int(end)])
        except:
            span_text.append('NaN')
    return ', '.join(span_text)

def span_len(span_text):
    # return the number of words in the span
    return len(span_text.split(' '))

def main(args):

    type_count = {}
    span_len_count = {}

    with open(args.outputfile, 'w') as fw:

        with open(args.inputfile, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)

            for line in reader:
                text = line[1]
                type = line[1].split(':')[0]
                hpo = line[2]
                span_text = process_spans(line[4], text)

                if type not in type_count:
                    type_count[type] = 1
                else:
                    type_count[type] += 1

                if span_len(span_text) not in span_len_count:
                    span_len_count[span_len(span_text)] = 1
                else:
                    span_len_count[span_len(span_text)] += 1

                fw.write(f'{type}||{span_text}||{hpo}\n')

    # sort the dictionary by key
    span_len_count = dict(sorted(span_len_count.items(), key=lambda item: item[0]))

    print('Done!')
    print('----------------type_count----------------')
    print(type_count)
    print('--------------span_len_count--------------:')
    print(span_len_count)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputfile', type=str, default='./BioCreative VIII Track 3/BioCreativeVIII3_ValSet.tsv')
    parser.add_argument('--outputfile', type=str, default='./dataset/val_annotation.txt')
    args = parser.parse_args()
    main(args)
