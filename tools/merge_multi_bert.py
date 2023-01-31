import sys, os
sys.path.insert(1, os.path.abspath("."))
sys.path.insert(1, os.path.abspath(".."))


from model.MultiModalBert import merge_multi_modal_bert

if __name__ == '__main__':
    merge_multi_modal_bert()
