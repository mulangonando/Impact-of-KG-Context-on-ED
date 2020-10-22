with open('dev.tsv') as f:
    for counter, line in enumerate(f):
        print(line.split("\t"))
        break
