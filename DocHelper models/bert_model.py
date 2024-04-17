import numpy as np
import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
import PyPDF2
import os


#Initilaize the mModel
def initialize_model(model_path=None):
    if model_path:
        os.chdir(model_path)
    model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    return model, tokenizer


# Defining function for pdf search
def find_answer_pdf(question, pdf_path, model, tokenizer):
    if not os.path.exists(pdf_path):
        raise Exception('PDF file does not exist!')

    def find_factors(x):
        x = x - 1
        factor = x
        for i in range(1, x + 1):
            if x % i == 0:
                if not i > 512:
                    factor = i
        if factor == 1:
            factor, x = find_factors(x)
        return factor, x

    # Initialize the answer variable
    answer = ''

    # Read PDF for the context
    pdf_file_object = open(pdf_path, 'rb')
    pdf_reader = PyPDF2.PdfReader(pdf_file_object)
    count = len(pdf_reader.pages)

    for num in range(count):
        if len(answer) > 0:
            break
        print(f'Searching Page {num + 1}/{count}')
        page = pdf_reader.pages[num]
        temp = page.extract_text()
        if len(temp.replace('.', '').replace('\n', '')) > 50:
            # Clean the text from PDF
            text = temp.strip()
            text = text.replace('\n', '')
            # If index page, we get too many fullstops. Skip such pages
            if text.count('.') / len(text) > .25:
                continue

            # Convert the tokens to ids using encode function of tokenizer
            input_ids = tokenizer.encode(question, text, max_length=512, truncation=True)
            tokens = tokenizer.convert_ids_to_tokens(input_ids)

            temp = np.array(tokens)
            if len([x for x in temp if '#' in x]) / len(temp) > .5:
                continue

            # Split the tokens into half
            if len(tokens) > 450:
                factor, len_ = find_factors(len(tokens))
                temp = list(tokens[:len_ - 1])
                temp.append(tokens[-1])
                tokens = temp
                del temp
                temp = list(input_ids[:len_ - 1])
                temp.append(input_ids[-1])
                input_ids = temp
                del temp
                input_ids = np.array(input_ids).reshape((-1, factor))
                tokens = np.array(tokens).reshape((-1, factor))
            else:
                tokens = [tokens]
                input_ids = [input_ids]
            sep_idx = 0
            for i in range(len(input_ids)):
                # Since we are splitting the tokens, we need to append the question in the beginning and add separator at the end
                temp_tok = list(tokens[i])
                temp_id = list(input_ids[i])

                if i > 0:
                    temp_id = list(input_ids[0][:sep_idx + 1])
                    temp_id.extend(list(input_ids[i]))
                    temp_tok = list(tokens[0][:sep_idx + 1])
                    temp_tok.extend(list(tokens[i]))
                if tokens[i][-1] != '[SEP]':
                    temp_id.append(input_ids[0][sep_idx])
                    temp_tok.append(tokens[0][sep_idx])

                # first occurrence of [SEP] token to identify the question
                sep_idx = temp_id.index(tokenizer.sep_token_id)
                num_seg_a = sep_idx + 1
                num_seg_b = len(temp_id) - num_seg_a
                segment_ids = [0] * num_seg_a + [1] * num_seg_b  # making sure that every input token has a segment id
                assert len(segment_ids) == len(temp_id)

                # token input_ids to represent the input and token segment_ids to differentiate our segments - question and text
                output = model(torch.tensor([temp_id]), token_type_ids=torch.tensor([segment_ids]))

                # tokens with the highest start and end scores
                answer_start = torch.argmax(output.start_logits)
                answer_end = torch.argmax(output.end_logits)
                if answer_end >= answer_start:
                    ans = " ".join(temp_tok[answer_start:answer_end + 1])
                    if '[' not in ans.capitalize().strip() and len(ans.capitalize().strip()) != 0:
                        answer += ans.capitalize().replace(' ##', '')

    if len(answer) == 0:
        answer = "I am unable to find the answer to this question"
    return answer


if __name__ == '__main__':
    #Example
    question = 'Who moved up the avenue?'
    pdf_path = "B:\\Projects\\ChatWithPDF\\aftertwentyyearstext.pdf"
    model_path = 'B:\\Projects\\ChatWithPDF'

    # Initialize the model and tokenizer
    model, tokenizer = initialize_model(model_path)

    # Call the function with the initialized model and tokenizer
    answer = find_answer_pdf(question, pdf_path, model, tokenizer)

    print(f'Question : {question}')
    print(f' Answer  : {answer}')
