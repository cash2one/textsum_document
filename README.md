# install&Run
    As for how to install the package in testsum, please refer to source/testsum/README.md

# The program in testsum is used to generate abstract from article

# Input file
    As for the format of input file, please refer to source/testsum/data
    There are three files in /source/testsum/data, namely, vocal, data, test_data
## Please pay attention, the program only depends on vocal, data
## vocal
    It is the vocabulary file. 
    Each line contains two tokens split via space(empty space or tab). 
    The first token is the word and the second token is useless( we will not use it)
## data
    It is a binary file containing article and abstract
    The text file of data is shown in data_test. Please refer to source/testsum/data/text_data
## text_data
    It is a text file generated by data via program data_convert_example.py
    We can also generate binary file from text file via data_convert_example.py
    Each line is one instance. 
    One line is splited into several parts via tab. 
    Each parts is the format 'key=value'
    Each line must contains the key of article and the key of abstract. Other keys will not be used in the program
    For the value of article or abstract, <d> and </d> represents the document, <p> and </p> represents the paragraph, <s> and </s> represents the sentence

# Output
    As shown in source/testsum/README.md. The program will generate dir log_root log_root/train in training, /log_root/eval in evaluation and decode in decoding
    The final results of decoding will be shown in decode dir.
    The file with the name ref.. is the groundtruth
    The file with the name decode.. is the result


# Program Work flow
    As for training, please refer to picture/train.jpg. 
    As for test, please refer to picture/test.jpg.
    As for the module dependence, please refer to picture work_flow.jpg.
## work_flow
    seq2seq_attention.py is the main program. There are three model(train, eval , decoder). train model is training model. eval model is used to validation. decoder model is used to write the decode write.
    data.py is the program used to generate data
    batch_reader.py is the progam used to generate batch data
    seq2seq_lib.py stores some useful function
    seq2seq_attention_model.py is the program used to generate computing graph, as shown in picture/train.jpg and picture/test.jpg
    seq2seq_attention_decoder.py is the program used to write decoder result in decode model(namely test model)
    beam_search.py is used in decode model to program nbest results.

# API are shown in  [here](http://textsum-document.readthedocs.io/en/latest/modules.html)
# The picture for training ![](https://github.com/tzhongaa/textsum_document/blob/master/picture/train.jpg)
# The picture for test ![](https://github.com/tzhongaa/textsum_document/blob/master/picture/test.jpg)
# The picture for the work_flow ![](https://github.com/tzhongaa/textsum_document/blob/master/picture/work_flow.jpg)
