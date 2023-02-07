from macbert_corrector import MacBertCorrector

if __name__ == '__main__':
    nlp = MacBertCorrector("shibing624/macbert4csc-base-chinese").macbert_correct
    i = nlp('今天新情很好')
    print(i)